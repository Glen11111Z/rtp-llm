#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

class GroupSet;

// Batch-private KV blocks published by a streaming prefill before it enters
// RUNNING. Entries live outside the BlockTree: they are keyed by canonical cache
// key, scoped to one scheduler batch epoch, owned by the producing stream, and
// hold device-pool request references so the blocks cannot be recycled while
// registered.
//
// All methods are called with the owning BlockTreeCache mutex held.
class BatchPrivateBlockRegistry {
public:
    // Publishes keys[0..n) as batch-private entries. Re-registering a key for the
    // same owner is a no-op; a key already owned by another epoch is skipped so an
    // entry never mixes epochs. Returns the number of member blocks registered.
    size_t registerKeys(int64_t                                           epoch,
                        int64_t                                           owner_stream_id,
                        const CacheKeysType&                              keys,
                        const std::vector<std::vector<GroupSetResource>>& resources,
                        const std::vector<std::shared_ptr<GroupSet>>&     group_sets);

    // Removes every entry owned by (epoch, owner_stream_id) and releases the
    // registry reference of entries left without owners. Returns the number of
    // blocks released.
    size_t dropOwner(int64_t                                       epoch,
                     int64_t                                       owner_stream_id,
                     const std::vector<std::shared_ptr<GroupSet>>& group_sets);

    // Returns the per-group-set member blocks of a same-epoch entry, or nullptr
    // when the key is unknown to this epoch.
    const std::vector<std::vector<BlockIdxType>>* findBlocks(CacheKeyType key, int64_t epoch) const;

    // Drops entries whose epoch is older than keep_epoch. Entries are normally
    // released by dropOwner; this bounds leaks from an owner that never released.
    // Returns the number of blocks released.
    size_t reapStaleEpochs(int64_t keep_epoch, const std::vector<std::shared_ptr<GroupSet>>& group_sets);

    // Fallback bound for owners that never deregister: evicts whole entries in
    // ascending epoch order (newest epoch is kept) until the limit holds.
    // Returns the number of blocks released.
    size_t reclaimToLimit(size_t max_entries, const std::vector<std::shared_ptr<GroupSet>>& group_sets);

    size_t entryCount() const {
        return entries_.size();
    }
    size_t activeEpochCount() const;
    void   dump() const;

private:
    struct Entry {
        int64_t                                epoch{0};
        std::vector<std::vector<BlockIdxType>> blocks;  // [group_set_id][member_group]
        std::unordered_set<int64_t>            owners;
    };

    using OwnerKey = std::pair<int64_t, int64_t>;
    struct OwnerKeyHash {
        size_t operator()(const OwnerKey& key) const {
            return static_cast<size_t>(key.first) * 0x9E3779B97F4A7C15ULL ^ static_cast<size_t>(key.second);
        }
    };

    void releaseEntry(Entry& entry, const std::vector<std::shared_ptr<GroupSet>>& group_sets) const;
    // Erases one entry together with its owner back-pointers and registry refs.
    size_t eraseEntry(CacheKeyType key, std::unordered_map<CacheKeyType, Entry>::iterator entry,
                      const std::vector<std::shared_ptr<GroupSet>>& group_sets);
    static size_t entryBlockCount(const Entry& entry);

    std::unordered_map<CacheKeyType, Entry>                                       entries_;
    std::unordered_map<OwnerKey, std::unordered_set<CacheKeyType>, OwnerKeyHash> owner_keys_;
};

}  // namespace rtp_llm
