#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/batch_private/BatchPrivateBlockRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::makeStructuralDevicePool;

// Every test uses two FULL GroupSets with one device pool each, so a registered
// key holds one member block per GroupSet.
class BatchPrivateBlockRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (size_t group_set_id = 0; group_set_id < 2; ++group_set_id) {
            pools_.push_back(makeStructuralDevicePool(group_set_id, /*preallocate_all_blocks=*/false));
            group_sets_.push_back(std::make_shared<FullGroupSet>(
                std::vector<DeviceBlockPoolPtr>{pools_.back()}, nullptr, nullptr));
        }
        block_tree_cache_test::prepareGroupSetsForTest(group_sets_);
    }

    std::vector<std::vector<GroupSetResource>> makeResources(size_t key_count) {
        std::vector<std::vector<GroupSetResource>> resources(key_count);
        for (auto& per_key : resources) {
            per_key.resize(group_sets_.size());
            for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
                auto blocks = pools_[group_set_id]->malloc(1);
                RTP_LLM_CHECK(blocks.has_value());
                // Model the producer stream's own request reference.
                pools_[group_set_id]->incRef(*blocks);
                per_key[group_set_id].device_blocks.push_back(blocks->front());
            }
        }
        return resources;
    }

    size_t memberBlocks(const std::vector<std::vector<GroupSetResource>>& resources) const {
        size_t blocks = 0;
        for (const auto& per_key : resources) {
            for (const auto& resource : per_key) {
                blocks += resource.device_blocks.size();
            }
        }
        return blocks;
    }

    std::vector<uint32_t> refCounts(const std::vector<std::vector<GroupSetResource>>& resources) const {
        std::vector<uint32_t> counts;
        counts.reserve(memberBlocks(resources));
        for (const auto& per_key : resources) {
            for (size_t group_set_id = 0; group_set_id < per_key.size(); ++group_set_id) {
                const BlockIdxType block = per_key[group_set_id].device_blocks.front();
                counts.push_back(pools_[group_set_id]->refCount(block));
            }
        }
        return counts;
    }

    static std::vector<uint32_t> filled(size_t count, uint32_t value) {
        return std::vector<uint32_t>(count, value);
    }

    size_t referencedBlocks() const {
        size_t count = 0;
        for (const auto& pool : pools_) {
            count += pool->referencedBlocksNum();
        }
        return count;
    }

    BatchPrivateBlockRegistry       registry_;
    std::vector<DeviceBlockPoolPtr> pools_;
    std::vector<GroupSetPtr>        group_sets_;
};

TEST_F(BatchPrivateBlockRegistryTest, RegistersAndFindsSameEpochEntries) {
    const CacheKeysType keys{11, 12, 13};
    auto                resources = makeResources(keys.size());

    EXPECT_EQ(registry_.registerKeys(2, 7, keys, resources, group_sets_), memberBlocks(resources));
    EXPECT_EQ(registry_.entryCount(), keys.size());
    EXPECT_EQ(registry_.activeEpochCount(), 1u);
    EXPECT_EQ(refCounts(resources), filled(memberBlocks(resources), 2));

    for (size_t i = 0; i < keys.size(); ++i) {
        const auto* blocks = registry_.findBlocks(keys[i], 2);
        ASSERT_NE(blocks, nullptr);
        ASSERT_EQ(blocks->size(), group_sets_.size());
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            EXPECT_EQ((*blocks)[group_set_id], resources[i][group_set_id].device_blocks);
        }
    }
    EXPECT_EQ(registry_.findBlocks(keys.front(), 3), nullptr);
    EXPECT_EQ(registry_.findBlocks(keys.front(), 0), nullptr);
    EXPECT_EQ(registry_.findBlocks(99, 2), nullptr);
}

TEST_F(BatchPrivateBlockRegistryTest, RejectsEmptyMismatchedAndIncompleteResources) {
    const CacheKeysType keys{21};

    auto mismatched        = makeResources(1);
    auto wrong_group_count = makeResources(1);
    wrong_group_count.front().pop_back();
    auto null_block = makeResources(1);
    null_block.front().front().device_blocks.front() = NULL_BLOCK_IDX;
    auto missing_member = makeResources(1);
    missing_member.front().front().device_blocks.clear();
    auto usable = makeResources(1);

    const size_t refs_before = referencedBlocks();
    EXPECT_EQ(registry_.registerKeys(
                  2, 7, CacheKeysType{}, std::vector<std::vector<GroupSetResource>>{}, group_sets_),
              0u);
    EXPECT_EQ(registry_.registerKeys(2, 7, CacheKeysType{21, 22}, mismatched, group_sets_), 0u);
    EXPECT_EQ(registry_.registerKeys(2, 7, keys, wrong_group_count, group_sets_), 0u);
    EXPECT_EQ(registry_.registerKeys(2, 7, keys, null_block, group_sets_), 0u);
    EXPECT_EQ(registry_.registerKeys(2, 7, keys, missing_member, group_sets_), 0u);
    EXPECT_EQ(registry_.registerKeys(2, 7, keys, usable, {}), 0u);

    EXPECT_EQ(registry_.entryCount(), 0u);
    EXPECT_EQ(registry_.activeEpochCount(), 0u);
    EXPECT_EQ(referencedBlocks(), refs_before);
}

TEST_F(BatchPrivateBlockRegistryTest, ReRegisteringSameOwnerIsIdempotent) {
    const CacheKeysType keys{31};
    auto                resources = makeResources(keys.size());

    EXPECT_EQ(registry_.registerKeys(2, 7, keys, resources, group_sets_), memberBlocks(resources));
    EXPECT_EQ(registry_.registerKeys(2, 7, keys, resources, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 1u);
    EXPECT_EQ(refCounts(resources), filled(memberBlocks(resources), 2));

    EXPECT_NE(registry_.findBlocks(keys.front(), 2), nullptr);
    EXPECT_EQ(registry_.dropOwner(2, 7, group_sets_), memberBlocks(resources));
    EXPECT_EQ(registry_.entryCount(), 0u);
    EXPECT_EQ(refCounts(resources), filled(memberBlocks(resources), 1));
}

TEST_F(BatchPrivateBlockRegistryTest, SecondOwnerSharesEntryUntilLastDrop) {
    const CacheKeysType first_keys{41};
    auto                first_resources  = makeResources(first_keys.size());
    auto                second_resources = makeResources(first_keys.size());

    EXPECT_EQ(registry_.registerKeys(2, 7, first_keys, first_resources, group_sets_),
              memberBlocks(first_resources));
    EXPECT_EQ(registry_.registerKeys(2, 8, first_keys, second_resources, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 1u);
    // The entry keeps the first owner's blocks and its single registry reference.
    EXPECT_EQ(refCounts(first_resources), filled(memberBlocks(first_resources), 2));
    EXPECT_EQ(refCounts(second_resources), filled(memberBlocks(second_resources), 1));

    EXPECT_EQ(registry_.dropOwner(2, 7, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 1u);
    EXPECT_NE(registry_.findBlocks(first_keys.front(), 2), nullptr);

    EXPECT_EQ(registry_.dropOwner(2, 8, group_sets_), memberBlocks(first_resources));
    EXPECT_EQ(registry_.entryCount(), 0u);
    EXPECT_EQ(registry_.findBlocks(first_keys.front(), 2), nullptr);
    EXPECT_EQ(refCounts(first_resources), filled(memberBlocks(first_resources), 1));
}

TEST_F(BatchPrivateBlockRegistryTest, SkipsKeysOwnedByAnotherEpoch) {
    const CacheKeysType keys{51};
    auto                first_resources  = makeResources(keys.size());
    auto                second_resources = makeResources(keys.size());

    EXPECT_EQ(registry_.registerKeys(2, 7, keys, first_resources, group_sets_), memberBlocks(first_resources));
    EXPECT_EQ(registry_.registerKeys(3, 8, keys, second_resources, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 1u);
    EXPECT_EQ(registry_.findBlocks(keys.front(), 3), nullptr);
    const auto* blocks = registry_.findBlocks(keys.front(), 2);
    ASSERT_NE(blocks, nullptr);
    EXPECT_EQ((*blocks).front(), first_resources.front().front().device_blocks);

    EXPECT_EQ(registry_.dropOwner(3, 8, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 1u);
    EXPECT_EQ(registry_.dropOwner(2, 7, group_sets_), memberBlocks(first_resources));
    EXPECT_EQ(registry_.entryCount(), 0u);
}

TEST_F(BatchPrivateBlockRegistryTest, DropUnknownOwnerReturnsZero) {
    EXPECT_EQ(registry_.dropOwner(5, 5, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 0u);
}

TEST_F(BatchPrivateBlockRegistryTest, ReapStaleEpochsReleasesStaleEntries) {
    const CacheKeysType stale_keys{61, 62};
    const CacheKeysType fresh_keys{63};
    auto                stale_resources = makeResources(stale_keys.size());
    auto                fresh_resources = makeResources(fresh_keys.size());

    EXPECT_EQ(registry_.registerKeys(2, 7, stale_keys, stale_resources, group_sets_),
              memberBlocks(stale_resources));
    EXPECT_EQ(registry_.registerKeys(5, 8, fresh_keys, fresh_resources, group_sets_),
              memberBlocks(fresh_resources));

    EXPECT_EQ(registry_.reapStaleEpochs(0, group_sets_), 0u);
    EXPECT_EQ(registry_.reapStaleEpochs(5, group_sets_), memberBlocks(stale_resources));
    EXPECT_EQ(registry_.entryCount(), fresh_keys.size());
    EXPECT_EQ(registry_.findBlocks(stale_keys.front(), 2), nullptr);
    EXPECT_NE(registry_.findBlocks(fresh_keys.front(), 5), nullptr);
    EXPECT_EQ(refCounts(stale_resources), filled(memberBlocks(stale_resources), 1));
    EXPECT_EQ(refCounts(fresh_resources), filled(memberBlocks(fresh_resources), 2));

    EXPECT_EQ(registry_.dropOwner(5, 8, group_sets_), memberBlocks(fresh_resources));
    EXPECT_EQ(registry_.entryCount(), 0u);
}

TEST_F(BatchPrivateBlockRegistryTest, ReclaimToLimitEvictsOldestEpochs) {
    const CacheKeysType old_keys{71, 72};
    const CacheKeysType new_keys{73, 74};
    auto                old_resources = makeResources(old_keys.size());
    auto                new_resources = makeResources(new_keys.size());

    EXPECT_EQ(registry_.registerKeys(2, 7, old_keys, old_resources, group_sets_), memberBlocks(old_resources));
    EXPECT_EQ(registry_.registerKeys(3, 8, new_keys, new_resources, group_sets_), memberBlocks(new_resources));

    EXPECT_EQ(registry_.reclaimToLimit(0, group_sets_), 0u);
    EXPECT_EQ(registry_.reclaimToLimit(2, group_sets_), memberBlocks(old_resources));
    EXPECT_EQ(registry_.entryCount(), new_keys.size());
    EXPECT_EQ(registry_.activeEpochCount(), 1u);
    EXPECT_EQ(registry_.findBlocks(old_keys.front(), 2), nullptr);
    EXPECT_NE(registry_.findBlocks(new_keys.front(), 3), nullptr);
    EXPECT_EQ(refCounts(old_resources), filled(memberBlocks(old_resources), 1));

    EXPECT_EQ(registry_.dropOwner(3, 8, group_sets_), memberBlocks(new_resources));
}

TEST_F(BatchPrivateBlockRegistryTest, ReclaimToLimitKeepsNewestEpochAboveLimit) {
    const CacheKeysType keys{81, 82};
    auto                resources = makeResources(keys.size());

    EXPECT_EQ(registry_.registerKeys(3, 8, keys, resources, group_sets_), memberBlocks(resources));
    EXPECT_EQ(registry_.reclaimToLimit(1, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), keys.size());
    EXPECT_NE(registry_.findBlocks(keys.front(), 3), nullptr);

    EXPECT_NE(registry_.dropOwner(3, 8, group_sets_), 0u);
    EXPECT_EQ(registry_.entryCount(), 0u);
}

TEST_F(BatchPrivateBlockRegistryTest, DumpIsSafeForLiveAndEmptyRegistry) {
    registry_.dump();
    const CacheKeysType keys{91};
    auto                resources = makeResources(keys.size());
    EXPECT_EQ(registry_.registerKeys(2, 7, keys, resources, group_sets_), memberBlocks(resources));
    EXPECT_NO_THROW(registry_.dump());
}

}  // namespace
}  // namespace rtp_llm
