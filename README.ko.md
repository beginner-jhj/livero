# livero

[English](README.md) | 한국어

C로 작성한, 의존성 없는 온디바이스 벡터 데이터베이스입니다.

livero는 레코드 — 키, 값, 선택적 벡터, 그리고 타입이 있는 메타데이터 필드 —
를 저장하고, 필터가 결합된 최근접 이웃(nearest-neighbor) 쿼리를 전부
온디바이스에서 처리합니다. LSM-tree 스토리지 엔진(WAL + memtable + SST)과 HNSW
벡터 인덱스를 결합했고, 모든 기능을 문자열 기반의 작은 API로 노출해 FFI(JNI /
Swift 등)로 바인딩하기 쉽습니다.

> 상태: v1. 동작하고 테스트되어 있으며, 아직 최적화되지 않은 경로들이 있습니다
> (로드맵 참고). 현재 ARM64 전용(Apple Silicon / ARM Linux)입니다 — ARM NEON
> SIMD를 사용하며 아직 x86 폴백이 없습니다.

## 목차

- [왜 만들었나](#왜-만들었나)
- [특징](#특징)
- [동작 방식](#동작-방식)
- [사용법 (C)](#사용법-c)
- [빌드](#빌드)
  - [파이썬 바인딩](#파이썬-바인딩-선택)
- [벤치마크](#벤치마크)
- [알려진 한계 / 로드맵](#알려진-한계--로드맵)
- [프로젝트에 대하여](#프로젝트에-대하여)
- [라이선스](#라이선스)

## 왜 만들었나

대부분의 벡터 데이터베이스는 서버를 전제합니다 — 실행할 프로세스, 네트워크 왕복,
기기를 떠나는 데이터. livero는 그 반대입니다. 앱에 링크해서 모든 것을 로컬에서
돌리는 라이브러리입니다. 목표는 데이터를 기기에 남겨두는 온디바이스 검색(RAG,
시맨틱 검색, 추천)이며, 서버도 런타임 의존성도 필요 없습니다.

## 특징

- **LSM-tree 스토리지** — write-ahead log, 인메모리 skiplist memtable, 컴팩션이
  있는 불변 정렬 SST.
- **HNSW 벡터 인덱스** — 약 O(log N)의 근사 최근접 이웃 검색.
- **ARM NEON SIMD** 거리 커널 — `float32`와 `int8` 벡터 모두 지원하며, L2와
  내적(dot) 메트릭 제공.
- **문자열 기반 쿼리 API** — 필터가 평범한 문자열(`"age > 30 AND city == 'NYC'"`)
  이라, FFI로 바인딩할 때 구조체 마샬링이 필요 없습니다.
- **외부 의존성 제로** — C 표준 라이브러리와 POSIX만 사용합니다.

## 동작 방식

livero는 쿼리 시점에 만나는 두 축으로 이루어져 있습니다.

**스토리지는 LSM-tree입니다.** 쓰기는 먼저 write-ahead log로 가고(크래시
내구성을 위해), 그다음 인메모리 skiplist memtable에 들어갑니다. memtable이 차면
불변 정렬 SST 파일로 flush되고, 이후의 flush는 기존 SST와 병합됩니다(컴팩션).
읽기는 memtable을 먼저 확인하고(최신 버전), 없으면 SST로 넘어갑니다. 레코드는
시퀀스 번호로 버전이 매겨지므로, 업데이트와 삭제는 이전 것을 대체하는 더 새로운
레코드일 뿐입니다.

**벡터 검색은 HNSW 인덱스입니다.** 각 벡터는 계층 그래프의 노드가 되고, 검색은
모든 벡터를 훑는 대신 그래프를 탐욕적으로 따라가며 약 `O(log N)`에 근사 최근접
이웃을 찾습니다. 거리는 ARM NEON SIMD 커널로 계산합니다(`float32` / `int8`,
L2 / dot).

**둘을 잇는 다리 — `vector_index.lv`.** HNSW는 가장 가까운 벡터를
`vector_id`로 돌려주지만, 레코드 자체는 키로 정렬된 SST에 있습니다. 이걸 키
인덱스에서 하나씩 찾으면 히트당 `O(log N)`이고 — 게다가 키도 없고 `vector_id`만
있습니다. 그래서 livero는 `uint64` 레코드 오프셋의 평평한 배열인 동반 파일을
둡니다. vector_id가 순차적으로 할당되기 때문에, 레코드 `R`의 오프셋은 바이트
`vector_id * 8` 위치에 있고, 단 한 번의 `lseek(vector_id * 8)`로 SST
오프셋을 `O(1)`에 얻습니다:

```
HNSW 히트 → vector_id → vector_index.lv에서 lseek(vector_id * 8) → SST 오프셋 → 레코드
```

HNSW 그래프 자체는 저장하지 않습니다 — 원본 벡터만 저장합니다(`vectors.lv`).
열 때 livero는 레코드를 재생해 그래프를 메모리에 다시 만듭니다. 재구성 시간을
치르는 대신 디스크 포맷을 단순하게 유지하는 선택입니다.

## 사용법 (C)

```c
#include "livero.h"

// 스키마: 384차원 float32 벡터, L2 메트릭, 정수 메타데이터 필드 하나.
LVMetaFieldDef fields[] = {
    { .name = "year", .type = LV_META_INT },
};

Livero* db;
lv_create(&db, "mydb", /*flush_threshold*/ 1024,
          /*dim*/ 384, LV_VEC_FLOAT32, LV_METRIC_L2,
          /*field_count*/ 1, fields);

// 레코드 삽입: 키, 값, 벡터, 메타데이터 필드.
LVMetaField year = { .name = "year", .type = LV_META_INT, .value.i64 = 2024 };
lv_put(db, "doc-1", 5, "hello", 5, my_vector, 1, &year);

// 유사도로 정렬하고 필드로 필터링하는 최근접 이웃 쿼리.
// 벡터 검색은 ORDER BY "vector"로 트리거됩니다. 필터 문자열은 비어 있으면 안 되므로,
// 스키마 필드에 대한 실제 조건을 사용합니다.
LVQueryOption option = {
    .flags        = LV_QOPT_ORDER_BY | LV_QOPT_LIMIT,
    .limit        = 10,                 // 가장 가까운 10개 반환
    .order        = { .by = "vector", .dir = LV_ORDER_DESC },  // 높은 점수부터
    .vector_metric = LV_METRIC_L2,
};

LVQueryResultSet* results;
lv_query(db, "year >= 2020", query_vector, &option, &results);
for (LVSize32_t i = 0; i < results->size; i++) {
    // results->results[i].key / .value / .vector_score
}
lv_destroy_query_result_set(results);

lv_close(db);
```

점수는 `[0, 1]` 범위로 정규화되며(높을수록 더 유사), 따라서
`LV_QOPT_SCORE_FILTER`의 임계값도 이 범위로 지정합니다.

전체 API는 [`include/livero.h`](include/livero.h)에 있습니다: `lv_create`,
`lv_open`, `lv_put`, `lv_get`, `lv_update_value` / `_vector` / `_field`,
`lv_delete`, `lv_query`, `lv_close`.

## 빌드

요구 사항: CMake ≥ 3.15, C11 컴파일러, ARM64 플랫폼.

```sh
mkdir build && cd build
cmake ..
make
```

정적 라이브러리 `liblivero.a`와 C 단위 테스트 실행 파일(`test_arena`,
`test_vector`)이 빌드되며, 직접 실행할 수 있습니다:

```sh
./test_arena
./test_vector
```

기본 빌드는 AddressSanitizer가 켜진 Debug입니다. 빠른 빌드를 원하면 Release로
설정하세요:

```sh
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### 파이썬 바인딩 (선택)

파이썬 바인딩(통합 테스트 스위트와 벤치마크에 사용)은 CFFI로 빌드합니다:

```sh
pip install cffi
python build_livero.py            # _livero_cffi 확장 모듈 빌드
python -m pytest tests/           # 통합 테스트 스위트 실행
python tests/benchmark.py         # 벤치마크 실행
```

## 벤치마크

파이썬 CFFI 바인딩을 통해 최적화되지 않은 v1 빌드에서 측정한 간단한
기준선입니다. 튜닝된 수치가 아니라, 오늘 livero가 대략 어느 정도인지와 앞으로
발전할 여지를 보여주는 값입니다. `dim = 384`, `k = 10`, L2 메트릭.

| N        | insert/s   | query p50 | query p90 | recall@10 |
|----------|------------|-----------|-----------|-----------|
| 1,000    | ~1,600/s   | 0.088 ms  | 0.092 ms  | 99.8%     |
| 10,000   | ~1,040/s   | 0.171 ms  | 0.176 ms  | 75.5%     |

참고:
- 쿼리 지연 시간과 recall은 HNSW 인덱스의 성능을 반영합니다. 384차원에서
  0.2ms 미만의 최근접 이웃 쿼리가 핵심입니다.
- recall은 HNSW의 속도/정확도 트레이드오프입니다 — 작은 규모에서는 거의
  정확하고, 고차원에서 기본 탐색 폭으로는 낮아집니다. 탐색 폭(EF)을 키우면
  올라갑니다.
- 삽입 처리량은 바인딩의 파이썬↔C 마샬링이 지배적입니다. 순수 C는 더 빠릅니다.
  또한 HNSW 그래프 구성은 검색보다 본질적으로 더 무겁습니다.

## 알려진 한계 / 로드맵

livero는 v1입니다. 아직 최적화되지 않았거나 미완인 부분들을, 대략 우선순위
순으로 적습니다:

- **x86 지원** — 현재 ARM NEON 전용입니다. x86에서 빌드·실행되도록 하는 스칼라
  폴백이 다음 순서입니다.
- **동시성** — 현재 livero는 단일 writer이며 여러 곳이 스레드 안전하지 않습니다.
  제대로 된 동시성이 주요 방향입니다: 처리량을 위한 세밀한 스레드 안전성, 그리고
  flush/컴팩션의 백그라운드 실행(지금은 flush가 writer에서 인라인으로 일어나며,
  백그라운드 스레드에서 돌지 않습니다).
- **모바일 FFI** — 문자열 기반 API 전체가 livero를 JNI / Swift로 깔끔하게
  바인딩하기 위해 존재합니다. 안드로이드 / iOS 바인딩을 실제로 연결하고
  테스트하는 것 — 원래 이 프로젝트의 목적 — 이 우선순위입니다.
- **필터 없는 벡터 검색** — 현재 벡터 검색은 비어 있지 않은 필터 문자열을
  요구합니다. 필터 없는 순수 최근접 검색을 계획하고 있습니다.
- **내적(dot) recall** — dot은 단위 정규화된 벡터를 가정하므로, 정규화되지 않은
  입력은 recall을 떨어뜨립니다. 정규화 처리를 계획하고 있습니다.
- **쿼리별 탐색 폭(EF)** — 현재 EF는 쿼리로부터 유도되며 직접 설정하지
  않습니다. 일급 `search_ef` 옵션을 계획하고 있습니다.
- **고차원 recall** — 그래프 파라미터(M, EF_construction)가 아직 고차원에
  맞춰 튜닝되지 않았습니다.
- **크래시 복구 테스트 커버리지** — WAL torn-write 처리는 있지만, 명시적인
  크래시/절단 테스트를 계획하고 있습니다.
- **삽입 성능** — SIMD 경로 튜닝, 재사용 가능한 검색 버퍼, SST 인덱스 블록의
  이진 탐색.
- **VLA 정책** — 몇몇 내부 버퍼가 스택 VLA입니다. 검증 순서와 안전을 위해
  힙으로 옮기는 정리를 계획하고 있습니다.

## 프로젝트에 대하여

livero는 4개월 전, 데이터베이스 엔진이 실제로 어떻게 동작하는지 궁금해서, 직접
만들어보며 배우고 싶어서 시작했습니다. 처음에는 장난감이었습니다 — LSM-tree
키-값 스토어. 거기에 벡터 레이어가 붙고, HNSW 인덱스가 붙고, SIMD 커널이
붙었습니다. 질문 하나씩 따라가면서.

어느 순간부터 단순한 연습이 아니게 되었습니다. 내가 무엇을 만들고 있는지 더 깊이
이해할수록 더 애정이 생겼고, 그만큼 더 진지해졌습니다. livero는 그 결과입니다 —
계속 키워가고 싶은, 작고 의존성 없는 벡터 데이터베이스입니다. 더 나은 recall,
x86 지원, 제대로 된 패키징, 그리고 결국은 이 프로젝트가 처음부터 목표했던 모바일
바인딩까지.

이 중 무엇이든 흥미롭게 느껴진다면, 기여는 언제나 환영입니다. 재미있게 생각되는
부분이 있다면, 함께 만들어요.

## 라이선스

MIT — [LICENSE](LICENSE) 참고.

이름은 *libero*(이탈리아어로 "자유로운") + *vector*입니다: 서버로부터 자유롭고,
의존성으로부터 자유롭고, 기기 위에서 자유롭게 돌아가는 벡터 스토어.