# livero

[English](README.md) | **한국어**

livero는 온디바이스 생태계를 목표로 하는 작은 벡터 데이터베이스입니다.
순수 C로 작성되었고, 고정된 48KiB RAM만 사용하며, 인덱스 없이 전수조사로 검색합니다.

<p align="center">
  <img src="docs/assets/demo.gif" alt="livero demo: 강아지/고양이 사진 검색" width="700">
</p>

<p align="center">
  <sub>CLIP 임베딩으로 강아지/고양이 사진 1,000장을 검색하는 데모입니다. ✓는 fp32 전수조사와 같은 결과를 뜻합니다.</sub>
</p>

---

- [livero](#livero-1)
- [검색 품질](#검색-품질)
- [벤치마크](#벤치마크)
- [용량](#용량)
- [빌드](#빌드)
- [사용법](#사용법)
- [구조](#구조)
- [제한 사항](#제한-사항)
- [로드맵](#로드맵)

---

## livero

livero가 목표로 하는 온디바이스 환경에서 데이터 규모는 수천~수만 개 정도로 작습니다.
이 규모에서는 복잡한 ANN 인덱스 대신, 단순하고 예측 가능한 전수조사를 선택했습니다.

- **인덱스 없음**: 그래프나 클러스터를 메모리에 올릴 필요가 없습니다. 쿼리 시간은 데이터 크기에 비례하고, 검색 품질은 데이터 분포나 파라미터에 따라 흔들리지 않습니다.
- **고정 메모리**: 실행 중 동적 할당 없이 48KiB만 사용합니다.
- **삼진 양자화**: 벡터를 {-1, 0, +1}로 저장합니다. 384차원 fp32 벡터(1,536B)가 96B가 되어, 읽어야 할 데이터가 16분의 1로 줄어듭니다.
- **popcount 기반 dot 연산**: 곱셈 없이 비트 연산과 popcount만으로 내적을 계산합니다.

사전 실험에서 768차원 ternary 벡터 1M개의 dot 연산이 C 스칼라 코드로 6ms였습니다(메모리 내, 계산만 측정). 계산 속도는 병목이 되지 않는다고 판단했습니다.

## 검색 품질

ternary 양자화가 검색 품질을 얼마나 유지하는지 측정했습니다.
AG News 10,000개 문서, 쿼리 500개 기준으로, fp32 전수조사 결과를 정답으로 한 recall@10입니다.

| 임베딩 모델 | 차원 | recall@10 |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.7684 |
| bge-base-en-v1.5 | 768 | 0.7878 |

실험 코드: [`tests/python/recall_test.py`](tests/python/recall_test.py)

## 벤치마크

현재는 Mac과 LG Gram에서 실행한 벤치마크 결과만 있습니다.
모바일, SBC, MCU 등 더 제약적인 온디바이스 환경에서의 실험은 준비 중입니다.

**조건**: Release 빌드, 384차원, 값 길이 487B, top-k 10, GET/QUERY는 각 100회 측정

### Mac (Apple M5)

| 레코드 수 | PUT 평균/ p50 / p99 | GET 평균/ p50 / p99 | QUERY 평균 / p50 / p99 |
|---|---|---|---|
| 1,000 | 0.102 ms / 0.090 ms / 0.221 ms| 0.003 ms / 0.003 ms / 0.004 ms | 0.028 ms / 0.028 ms / 0.044 ms|
| 10,000 | 0.074 ms / 0.073 ms / 0.098 ms | 0.004 ms / 0.004 ms / 0.011 ms | 0.097 ms / 0.095 ms / 0.152 ms |
| 100,000 | 0.082 ms  / 0.081 ms /  0.117 ms | 0.006 ms / 0.006 ms / 0.011 ms | 0.713 ms / 0.696 ms / 1.338 ms |


### LG Gram (Intel i7-1260P, Fedora)

x86-64에서는 기본으로` -mpopcnt`가 켜집니다(`-DLV_ENABLE_POPCNT=OFF`로 끌 수 있음)

| 레코드 수 | PUT 평균 / p50 / p99 | GET 평균/ p50 / p99 | QUERY 평균/ p50 / p99 |
|---|---|---|---|
| 1,000 | 6.631 ms / 6.606 ms / 10.786 ms | 0.011 ms / 0.012 ms / 0.028 ms | 0.099 ms / 0.095 ms / 0.274 ms |
| 10,000 | 6.910 ms / 6.781 ms /  10.913 ms | 0.010 ms / 0.012 ms / 0.028 ms | 0.262 ms / 0.249 ms / 0.582 ms |
| 100,000 | 6.789 ms / 6.672 ms / 10.809 ms | 0.007 ms / 0.007 ms / 0.015 ms | 1.897 ms / 1.739 ms / 2.989 ms |

**PUT 시간이 두 환경에서 크게 다른 이유**: livero는 모든 파일을 `O_SYNC`로 엽니다. Linux에서는 `O_SYNC`가 저장장치까지 실제로 기록하므로 PUT 하나에 약 6.7ms가 걸립니다. macOS에서는 `O_SYNC`가 드라이브 캐시까지 비우지 않기 때문에 훨씬 빠르게 측정됩니다(`F_FULLFSYNC`가 필요합니다). 따라서 Mac의 PUT 수치는 Linux와 같은 수준의 내구성을 뜻하지 않습니다.

벤치마크 코드: [`tests/benchmark.c`](tests/benchmark.c)

## 용량

최대 디스크 크기를 정해 두고 가득 찰 때까지 PUT한 결과입니다(fp32 입력, 4MiB).
레코드당 메타데이터는 23B로 일정합니다.

| 차원 | 값 길이 | 저장된 레코드 수 | 레코드당 크기 | 디스크 사용률 |
|---|---|---|---|---|
| 128 | 1B | 74,897 | 56.0 B | 100.0% |
| 384 | 1B | 34,952 | 120.0 B | 100.0% |
| 768 | 1B | 19,417 | 216.0 B | 100.0% |
| 384 | 100B | 19,151 | 219.0 B | 100.0% |
| 384 | 1–487B (무작위) | 11,567 | 362.6 B | 99.99% |
| 768 | 1–487B (무작위) | 9,155 | 458.1 B | 99.99% |

재오픈 후 레코드 수는 모든 조건에서 일치했습니다.

실험 코드: [`tests/python/capacity_test.py`](tests/python/capacity_test.py)

## 빌드

C23을 지원하는 컴파일러와 CMake가 필요합니다.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
ctest --test-dir build-release
```

`CMAKE_BUILD_TYPE`을 지정하지 않으면 AddressSanitizer가 켜진 Debug 빌드가 됩니다. 성능을 측정할 때는 꼭 Release로 빌드하세요.

x86-64에서는 하드웨어 `POPCNT` 명령어를 사용하도록 `-mpopcnt`로 빌드합니다. 이 플래그가 없으면 쿼리가 약 2.5배 느려집니다.

## 사용법

### C

```c
#include <sys/stat.h>
#include "livero.h"

int main(void) {
    // The directory must exist before lv_create.
    mkdir("./my_db", 0755);

    const LVConfig config = {
        .max_disk_bytes   = 4 << 20,              // 4 MiB
        .vector_type      = LV_VEC_TYPE_FP32,
        .vector_dimension = 384,
    };

    Livero* db = NULL;
    if (lv_create("./my_db", &config, &db) != LV_OK) return 1;

    float vector[384] = { /* embedding */ };
    const LVDate date = { .year = 2026, .month = 10, .day = 15 };
    const char* value = "hello";
    LVSeq_t key;
    lv_put(db, vector, &date, (const uint8_t*)value, 5, &key);

    // Top-10 search. Pass NULL dates to search all records.
    LVQueryResult* results = NULL;
    LVSize_t count = 0;
    lv_query(db, 10, vector, NULL, NULL, &results, &count);

    lv_close(db);
    return 0;
}
```


### Python

cffi 바인딩을 제공합니다.

```bash
pip install cffi
python tests/python/build.py
```

```python
#run in tests/python
from livero import Livero, LVStatus, LVVectorType, LVDate

db = Livero()
db.create("./my_db", {
    "max_disk_bytes": 4 << 20,
    "vector_type": LVVectorType.LV_VEC_TYPE_FP32,
    "vector_dimension": 384,
})

status, key = db.put(vector, LVDate(2026, 10, 15, 0, 0, 0), b"hello")
status, results = db.query(vector, 10, None, None)
for r in results:
    print(r.key, r.date, r.value)

db.close()
```

### API

| 함수 | 설명 |
|---|---|
| `lv_create` | 새 데이터베이스를 만듭니다. |
| `lv_open` | 기존 데이터베이스를 엽니다. |
| `lv_put` | 벡터, 날짜, 값을 저장하고 key를 돌려줍니다. |
| `lv_get` | key로 날짜와 값을 읽습니다. |
| `lv_delete` | key에 해당하는 레코드를 삭제합니다(tombstone). |
| `lv_query` | top-k 검색을 합니다. 날짜 범위로 필터링할 수 있습니다. |
| `lv_db_info` | 설정, 레코드 수, 파일 크기를 돌려줍니다. |
| `lv_close` | 데이터베이스를 닫습니다. |

- 최대 top-k: 64 (`LV_MAX_K`)
- 최대 값 길이: 487B (`LV_MAX_VALUE_LEN`)
- 최대 벡터 차원: 768 (`LV_MAX_VECTOR_DIMENSION`)
- 입력 벡터 타입: `fp32`, `int8` (저장 시 ternary로 양자화)

### 테스트

`ctest`로 다음을 검증합니다.

- PUT/GET 왕복
- 재오픈 후 데이터와 key 연속성 유지
- 삭제 후 GET 실패
- 쿼리 결과가 전수조사 정답과 일치
- 날짜 필터 경계값 (경계 포함, 결과 0개인 범위)
- 삭제된 레코드가 쿼리 결과에서 제외

## 구조

### 메모리 (48KiB)

| 영역 | 크기 |
|---|---|
| 쿼리 결과 저장소 | 504B × 64 = 31.5KiB |
| 파일 배치 버퍼 | 16KiB |
| `Livero` 구조체 | 64B |
| 여유 공간 | 448B |
| 합계 | 48KiB |

실행 중에는 추가로 메모리를 할당하지 않습니다.

### 파일

livero는 세 개의 파일로 데이터를 관리합니다. 모든 파일은 append-only입니다.

- `val_log.lv`: 원본 값(raw value)을 저장합니다. 최대 487B입니다.
- `vec_log.lv`: 양자화된 벡터를 저장합니다.
- `db.lv`: 데이터베이스 상태(디스크 사용량, 각 파일 메타데이터)와 레코드마다 key, date, value_len, val_log_offset을 저장합니다.

쓰기 순서는 `val_log.lv` → `vec_log.lv` → `db.lv`이고, `db.lv`의 헤더를 마지막에 기록합니다. 헤더가 기록된 시점이 커밋 지점이라, 중간에 전원이 꺼져도 헤더가 가리키는 범위까지만 유효한 데이터로 봅니다.

### 키

livero의 키는 추가 순서대로 증가하는 seq 값입니다.
커스텀 키는 사용할 수 없지만, 그 대신 레코드 크기를 고정할 수 있고 별도의 인덱스 없이 키 순서가 정렬되어 있음을 보장합니다. GET은 이 정렬을 이용해 이진 탐색으로 레코드를 찾습니다.

### 쿼리

1. 쿼리 벡터를 ternary로 양자화합니다.(bitnet 1.58B absmean 방식)
2. `vec_log.lv`를 16KiB씩 읽으며 각 벡터와 dot을 계산합니다.
3. 크기 k의 min-heap으로 상위 k개를 유지합니다. 힙이 가득 찼고 dot 값이 힙의 최솟값 이하이면 바로 건너뜁니다.
4. 힙에 들어갈 후보만 `db.lv`에서 날짜와 삭제 여부를 확인합니다. 스캔이 진행될수록 힙의 최솟값이 올라가므로, 실제로 확인하는 후보는 전체의 일부입니다.

### 삼진 dot 연산

각 벡터를 mask 비트(0이 아닌지)와 sign 비트(음수인지) 두 평면으로 저장합니다.

```
M   = mask_a & mask_b
X   = (sign_a ^ sign_b) & M
dot = popcount(M) - 2 * popcount(X)
```

## 제한 사항

- 단일 프로세스, 단일 스레드에서만 사용할 수 있습니다.
- 레코드 수정(update)을 지원하지 않습니다.
- 삭제는 tombstone 방식이며, 삭제된 공간을 회수하지 않습니다.
- 파일 오프셋이 4B라서 파일 하나의 크기는 4GiB를 넘을 수 없습니다.
- 커스텀 키를 사용할 수 없습니다.
- macOS에서는 `O_SYNC`가 드라이브 캐시까지 비우지 않으므로, Linux와 같은 수준의 내구성을 보장하지 않습니다.

## 로드맵

- 현재 dot 연산에는 `__builtin_popcountll`을 사용합니다. SIMD 커널을 추가할 계획입니다.
- 내구성을 보장하기 위해 파일을 `O_SYNC`로 열어서, Linux에서는 PUT 하나에 약 6.7ms가 걸립니다. 사용자가 내구성 수준을 선택할 수 있게 하고, 내부 캐시를 사용할 예정입니다.
