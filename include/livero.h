#ifndef LIVERO
#define LIVERO

#include "livero_types.h"

/*
* lv_create: Create a new database at the given path.

* Parameters:
* --db_path: A directory to create. Must not be NULL and already contain database.
* --config: Must not be NULL. (see LVConfig)
* --option: Must not be NULL. (see LVOption)
* --db_out: A new handle on success. Must not be NULL. (it's not NULL on failure.)

* Returns:
* --LV_OK: On success.
* --LV_ERR_INVALID_PARAM: A NULL argument, or a config/option contains any invalid argument.
* --LV_ERR_IO: File create/close failed.
* --LV_ERR_EXIST: The given directory already has database.
* --LV_ERR_ALLOCATE_MEMORY: Failed to create a new db on memory.
*/
LVStatus lv_create(const char* db_path, const LVConfig* config, const LVOption* option, Livero** db_out);

/*
* lv_open: Open a existing database in the given path.

* Parameters:
* --db_path: A directory to open. Must not be NULL. Must already have database.
* --option: Must not be NULL. (see LVOption)
* --db_out: A new handle on success. Must not be NULL. (it's not NULL on failure.)

* Returns:
* --LV_OK: On success.
* --LV_ERR_INVALID_PARAM: A NULL argument, or a config/option contains any invalid argument.
* --LV_ERR_IO: File open/close failed.
* --LV_ERR_NOT_EXIST: The given directory does not have database.
* --LV_ERR_ALLOCATE_MEMORY: Failed to create a new db on memory.
*/
LVStatus lv_open(const char* db_path, const LVOption* option, Livero** db_out);

/*
* lv_put: Put a new row/vector/value set to db.
*
* Parameters:
* --db: Open database. Must not be NULL.
* --vector: Raw vector data. Must not be NULL.
* --value: Opaque payload stored with a vector data, Livero never interprets its bytes.
        It's copied to the pending buffer, caller can reuse or free the buffer after function returns a status.
        Must not be NULL.
* --value_len: How many bytes the value are. Must be under max_value_len and greater than 0.
* --date: Time data stored with a vector and value. If it's NULL, this will be date at that time.
* --seq_out: This is the only key to handle this record (lv_get, lv_delete).
             Seq is never reused and only increased (later put gets a larger seq).
             Seqs are not contigous, you may see a gap after reopen (do not use seq to count records or guess other seqs).
             (*Seqs are assigned immediatly, but the record becomes durable when it is commited. 
             If db is crashed before commit, the record which the seq points to is lost.)
             If it's NULL, Livero will not assign seq to out parameter, you can still handle the row with lv_query.

* Returns:
* --LV_OK: On success.
* --LV_ERR_INVALID_PARAM: A NULL argument (db, vector, value). value_len is 0 or greaater than max_value_len. date is invalid.
* --LV_ERR_DISK_FULL: The total size of a new record and rows which are going to be flushed exceeds max_disk_bytes.
* --LV_ERR_IO: Commit failed.
*/
LVStatus lv_put(Livero* db, void* vector,void* value,const LVSize16_t value_len, const LVDate* date, LVSeq_t* seq_out);

/*
* lv_get: Get a date, value(if value buffer exists), value_len set.

* Parameters:
* --db: Open database. Must not be NULL.
* --target_seq: A seq key pointing to your target.
* --result: A buffer to contain your target. Must not be NULL. (see LVGetResult)

* Returns:
* --LV_OK: On success.
* --LV_ERR_INVALID_PARAM: A NULL argument.
* --LV_ERR_NOT_FOUND: Can't find record whose seq matches to the target_seq.
* --LV_ERR_DELETED: Record has a tombstone mark.
* --LV_ERR_IO: IO error happens during disk search.
*/
LVStatus lv_get(Livero* db, const LVSeq_t target_seq, LVGetResult* result);

/*
* lv_delete: Mark tombstone on the row of the target_seq.

* Parameters:
* --db: Open database. Must not be NULL.
* --target_seq: A seq key pointing to your target.

* Returns:
* --LV_OK: On success.
* --LV_ERR_INVALID_PARAM: A NULL argument.
* --LV_ERR_NOT_FOUND: Can't find record whose seq matches to the target_seq.
* --LV_ERR_IO: IO error happens during disk search or marking tombstone on the flushed target.
*/
LVStatus lv_delete(Livero* db, const LVSeq_t target_seq);

/*
* lv_query: Retrieve N LVQueryResult(s). (N <= k)

* Parameters:
* --db: Open database. Must not be NULL.
* --k: Top-k number of your query. Must be less than max_top_k.
* --query_vector: Raw vector to query. Must not be NULL.
* --option: An option which contains date filters and a use_postsearch flag. (see LVQueryOption).
            If it's NULL, Livero will use default query setting (No date filtering, use pre tombstone filtering).
* --result: A buffer to contain N LVQueryResult(s). Must not be NULL.
            Must be pre-allocated and its minimum size must be k*sizeof(LVQueryResult). (see LVQueryResult)
* --result_size: A number of assigned LVQueryResult(s). Must not be NULL.
                (This may be less than required k if there are not enough candidates, post-filter used, or date filter is too sharp.)

* Returns:
* --LV_OK: On success.
* --LV_ERR_INVALID_PARAM: A NULL parameter (db, query_vector, result, result_size).
* --LV_ERR_IO: IO error happens during reads ternary vectors or rows from disk.
                
*/
LVStatus lv_query(Livero* db,const LVTopK_t k, const void* query_vector, const LVQueryOption* option, LVQueryResult* result, LVSize32_t* result_size);

/*
* lv_close: Close all file buffers and free db memory arena.

* Parameter:
* --db: Open db. If it's NULL, lv_close warns that db is missing and early returns.

* lv_close returns nothing, only warns you if there is a fail while flush remaining pending recors and close files.
*/
void lv_close(Livero* db);

/*
* lv_stat: Assign LVDBStat to out paramerter.

* Parameters:
* --db: Open db. If it's NULL, lv_stat doesn't do anything.
* --stat: A buffer to contain database internal stats. If it's NULL, lv_stat doesn't do anything (see LVDBStat).

* lv_stat returns nothing, it assigns LVDBStat if both db and stat are not NULL.
*/
void lv_stat(const Livero* db, LVDBStat* stat);

#endif
