#include "rtp_llm/cpp/cache/block_tree_cache/batch_private/BatchPrivateBlockRegistry.h"

#include <algorithm>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool hasCompleteDeviceBlocks(const GroupSetResource& resource) {
    return !resource.device_blocks.empty()
           && std::all_of(resource.device_blocks.begin(), resource.device_blocks.end(), [](BlockIdxType block) {
                  return !isNullBlockIdx(block);
               });
}

MultiNodeResource deviceResource(size_t group_set_id, const std::vector<BlockIdxType>& blocks) {
    MultiNodeResource resource{group_set_id, Tier::DEVICE};
    resource.node_blocks.emplace_back(nullptr, blocks);
    return resource;
}

}  // namespace

size_t BatchPrivateBlockRegistry::registerKeys(int64_t                                           epoch,
                                               int64_t                                           owner_stream_id,
                                               const CacheKeysType&                              keys,
                                               const std::vector<std::vector<GroupSetResource>>& resources,
                                               const std::vector<std::shared_ptr<GroupSet>>&     group_sets) {
    RTP_LLM_CHECK(epoch > 0);
    if (keys.empty() || keys.size() != resources.size()) {
        RTP_LLM_LOG_WARNING("batch-private register rejected: keys=%zu resources=%zu", keys.size(), resources.size());
        return 0;
    }
    if (group_sets.empty()) {
        return 0;
    }
    for (const auto& per_key : resources) {
        if (per_key.size() != group_sets.size()) {
            RTP_LLM_LOG_WARNING("batch-private register rejected: group_set resources=%zu groups=%zu",
                                per_key.size(),
                                group_sets.size());
            return 0;
        }
        for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
            const auto& group_set = group_sets[group_set_id];
            if (group_set == nullptr
                || per_key[group_set_id].device_blocks.size() != group_set->devicePools().size()
                || !hasCompleteDeviceBlocks(per_key[group_set_id])) {
                RTP_LLM_LOG_WARNING("batch-private register rejected: incomplete device blocks at group_set=%zu",
                                    group_set_id);
                return 0;
            }
        }
    }

    const OwnerKey owner_key{epoch, owner_stream_id};
    auto&          owned_keys        = owner_keys_[owner_key];
    size_t         registered        = 0;
    size_t         registered_blocks = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheKeyType key   = keys[i];
        auto               entry = entries_.find(key);
        if (entry == entries_.end()) {
            Entry new_entry;
            new_entry.epoch = epoch;
            new_entry.blocks.reserve(resources[i].size());
            for (size_t group_set_id = 0; group_set_id < resources[i].size(); ++group_set_id) {
                new_entry.blocks.push_back(resources[i][group_set_id].device_blocks);
                group_sets[group_set_id]->referenceBlocks(deviceResource(group_set_id, new_entry.blocks.back()));
            }
            new_entry.owners.insert(owner_stream_id);
            registered_blocks += entryBlockCount(new_entry);
            entries_.emplace(key, std::move(new_entry));
            owned_keys.insert(key);
            ++registered;
            continue;
        }
        if (entry->second.epoch != epoch) {
            // A key first published by another batch keeps that batch's blocks; mixing
            // epochs would let a later batch observe a block set it never published.
            continue;
        }
        if (entry->second.owners.insert(owner_stream_id).second) {
            owned_keys.insert(key);
        }
    }
    if (owned_keys.empty()) {
        owner_keys_.erase(owner_key);
    }
    if (registered != 0) {
        RTP_LLM_LOG_DEBUG("batch-private epoch=%ld owner=%ld registered %zu/%zu keys, %zu blocks",
                          epoch,
                          owner_stream_id,
                          registered,
                          keys.size(),
                          registered_blocks);
    }
    return registered_blocks;
}

size_t BatchPrivateBlockRegistry::dropOwner(int64_t                                       epoch,
                                            int64_t                                       owner_stream_id,
                                            const std::vector<std::shared_ptr<GroupSet>>& group_sets) {
    auto owned = owner_keys_.find(OwnerKey{epoch, owner_stream_id});
    if (owned == owner_keys_.end()) {
        return 0;
    }
    const std::vector<CacheKeyType> keys(owned->second.begin(), owned->second.end());
    owner_keys_.erase(owned);
    size_t dropped        = 0;
    size_t dropped_blocks = 0;
    for (const CacheKeyType key : keys) {
        auto entry = entries_.find(key);
        if (entry == entries_.end() || entry->second.epoch != epoch) {
            continue;
        }
        entry->second.owners.erase(owner_stream_id);
        ++dropped;
        if (entry->second.owners.empty()) {
            dropped_blocks += entryBlockCount(entry->second);
            releaseEntry(entry->second, group_sets);
            entries_.erase(entry);
        }
    }
    if (dropped != 0) {
        RTP_LLM_LOG_DEBUG("batch-private epoch=%ld owner=%ld dropped %zu keys, %zu blocks, %zu entries remain",
                          epoch,
                          owner_stream_id,
                          dropped,
                          dropped_blocks,
                          entries_.size());
    }
    return dropped_blocks;
}

const std::vector<std::vector<BlockIdxType>>* BatchPrivateBlockRegistry::findBlocks(CacheKeyType key,
                                                                                    int64_t      epoch) const {
    const auto entry = entries_.find(key);
    if (entry == entries_.end() || entry->second.epoch != epoch) {
        return nullptr;
    }
    return &entry->second.blocks;
}

size_t BatchPrivateBlockRegistry::reapStaleEpochs(int64_t                                       keep_epoch,
                                                  const std::vector<std::shared_ptr<GroupSet>>& group_sets) {
    if (keep_epoch <= 0) {
        return 0;
    }
    std::vector<CacheKeyType> stale;
    for (const auto& [key, entry] : entries_) {
        if (entry.epoch < keep_epoch) {
            stale.push_back(key);
        }
    }
    if (stale.empty()) {
        return 0;
    }
    size_t reclaimed = 0;
    for (const CacheKeyType key : stale) {
        auto entry = entries_.find(key);
        if (entry == entries_.end()) {
            continue;
        }
        reclaimed += eraseEntry(key, entry, group_sets);
    }
    RTP_LLM_LOG_WARNING("batch-private reaped %zu entries older than epoch=%ld (%zu blocks); owners failed to "
                        "deregister",
                        stale.size(),
                        keep_epoch,
                        reclaimed);
    return reclaimed;
}

size_t BatchPrivateBlockRegistry::reclaimToLimit(size_t                                        max_entries,
                                                 const std::vector<std::shared_ptr<GroupSet>>& group_sets) {
    if (max_entries == 0 || entries_.size() <= max_entries) {
        return 0;
    }
    int64_t newest_epoch = 0;
    for (const auto& [key, entry] : entries_) {
        newest_epoch = std::max(newest_epoch, entry.epoch);
    }
    std::vector<std::pair<int64_t, CacheKeyType>> candidates;
    candidates.reserve(entries_.size());
    for (const auto& [key, entry] : entries_) {
        if (entry.epoch < newest_epoch) {
            candidates.emplace_back(entry.epoch, key);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    size_t reclaimed = 0;
    for (const auto& [epoch, key] : candidates) {
        if (entries_.size() <= max_entries) {
            break;
        }
        auto entry = entries_.find(key);
        if (entry == entries_.end()) {
            continue;
        }
        reclaimed += eraseEntry(key, entry, group_sets);
    }
    if (entries_.size() > max_entries) {
        RTP_LLM_LOG_WARNING("batch-private registry holds %zu entries above limit=%zu; the newest epoch alone "
                            "exceeds the bound, reclaimed_blocks=%zu",
                            entries_.size(),
                            max_entries,
                            reclaimed);
    } else {
        RTP_LLM_LOG_WARNING("batch-private registry evicted oldest epochs to %zu entries (limit=%zu), "
                            "reclaimed_blocks=%zu",
                            entries_.size(),
                            max_entries,
                            reclaimed);
    }
    return reclaimed;
}

size_t BatchPrivateBlockRegistry::activeEpochCount() const {
    std::unordered_set<int64_t> epochs;
    for (const auto& [key, entry] : entries_) {
        epochs.insert(entry.epoch);
    }
    return epochs.size();
}

void BatchPrivateBlockRegistry::dump() const {
    RTP_LLM_LOG_INFO("batch-private registry: entries=%zu active_epochs=%zu", entries_.size(), activeEpochCount());
    for (const auto& [key, entry] : entries_) {
        std::string owners;
        for (const int64_t owner : entry.owners) {
            if (!owners.empty()) {
                owners += ",";
            }
            owners += std::to_string(owner);
        }
        RTP_LLM_LOG_INFO("  key=%ld epoch=%ld owners=[%s] groups=%zu",
                         key,
                         entry.epoch,
                         owners.c_str(),
                         entry.blocks.size());
    }
}

size_t BatchPrivateBlockRegistry::eraseEntry(CacheKeyType                                       key,
                                             std::unordered_map<CacheKeyType, Entry>::iterator entry,
                                             const std::vector<std::shared_ptr<GroupSet>>&       group_sets) {
    const size_t blocks = entryBlockCount(entry->second);
    for (const int64_t owner : entry->second.owners) {
        auto owned = owner_keys_.find(OwnerKey{entry->second.epoch, owner});
        if (owned != owner_keys_.end()) {
            owned->second.erase(key);
            if (owned->second.empty()) {
                owner_keys_.erase(owned);
            }
        }
    }
    releaseEntry(entry->second, group_sets);
    entries_.erase(entry);
    return blocks;
}

size_t BatchPrivateBlockRegistry::entryBlockCount(const Entry& entry) {
    size_t blocks = 0;
    for (const auto& row : entry.blocks) {
        blocks += row.size();
    }
    return blocks;
}

void BatchPrivateBlockRegistry::releaseEntry(Entry&                                        entry,
                                             const std::vector<std::shared_ptr<GroupSet>>& group_sets) const {
    for (size_t group_set_id = 0; group_set_id < entry.blocks.size() && group_set_id < group_sets.size();
         ++group_set_id) {
        if (entry.blocks[group_set_id].empty() || group_sets[group_set_id] == nullptr) {
            continue;
        }
        group_sets[group_set_id]->unreferenceBlocks(deviceResource(group_set_id, entry.blocks[group_set_id]));
    }
}

}  // namespace rtp_llm
