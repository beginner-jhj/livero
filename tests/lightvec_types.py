from enum import Enum
from dataclasses import dataclass

#unit: byte
LV_META_NAME_MAX = 64 
LV_MAX_META_FIELDS = 32
LV_MAX_KEY_LEN = 1024
LV_MAX_VALUE_LEN = 2**24

class LVStatus(Enum):
    LV_OK = 2
    LV_QFILTER_T = 1
    LV_QFILTER_F = 0
    LV_ERR_IO = -1
    LV_ERR_OOM = -2
    LV_ERR_NOT_FOUND = -3
    LV_ERR_CORRUPT = -4
    LV_ERR_INVALID = -5
    LV_ERR_FULL = -6
    LV_ERR_DUPLICATE = -7
    LV_ERR_INVALID_DB = -8
    LV_ERR_INVALID_QUERY = -9
    LV_ERR_UNSUP_QOP = -10
    LV_ERR_EXISTS = -11

class LVVectorType(Enum):
    LV_VEC_FLOAT32 = 0
    LV_VEC_INT8 = 1

class LVMetaType(Enum):
    LV_META_STRING = 0
    LV_META_INT = 1
    LV_META_FLOAT = 2

class LVVectorMetric(Enum):
    LV_METRIC_L2 = 0
    LV_METRIC_DOT = 1

@dataclass
class lv_meta_field_value:
    i64:int = 0
    f64:float=0.0
    str_len:int = 0
    str_string:str = ""

@dataclass
class LVMetaFieldDef:
    name:str
    type:LVMetaType

@dataclass
class LVMetaField:
    name:str
    type:LVMetaType
    value:lv_meta_field_value


class LVQueryOptionFlag(Enum):
    LV_QOPT_NONE = 0
    LV_QOPT_LIMIT = 1
    LV_QOPT_ORDER_BY = 1<<1
    LV_QOPT_SCORE_FILTER = 1<<2

class LVQueryOrderDir(Enum):
    LV_ORDER_ASC = 0
    LV_ORDER_DESC = 1

class LVOrdbyType(Enum):
    LV_ORDBY_VEC = 0
    LV_ORDBY_FLOAT = 1
    LV_ORDBY_INT = 2
    LV_ORDBY_NONE = 3

class LVScoreBound(Enum):
    LV_SCORE_ABOVE = 0
    LV_SCORE_BELOW = 1

@dataclass
class LVOrdbyValue:
    score:float = 0.0
    f64:float = 0.0
    i64:int = 0

@dataclass
class LVQueryOption:
    flags:int
    limit:int
    top_k:int
    order_by:str
    order_dir:LVQueryOrderDir
    vector_score_filter_score:float
    vector_score_filter_bound:LVScoreBound
    vector_metric:LVVectorMetric

@dataclass
class LVQueryResult:
    node_seq:int
    vector_id:int
    key:bytes
    key_len:int
    value:bytes
    value_len:int
    vector_score:float

@dataclass
class LVQueryResultSet:
    size:int
    results:list[LVQueryResult]
