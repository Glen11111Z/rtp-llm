#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <mutex>
#include "autil/legacy/base64.h"

// token_num：每条约束路径的语义ID长度（token数量）
static const int token_num = 3;

// ---------------------------------------------------------------------------
// CsrLayerBaseIds：存储每层约束解码的 base token ID。
//
// extra_info (flat uint16) 中传递的值是 **偏移量** 而非绝对 token ID。
// 三元组 (offset_0, offset_1, offset_2) 对应的实际 token ID 为：
//   token_id_i = base_ids[i] + offset_i
//
// base token ID 来源于 tokenizer_config.json 中 <shop_0_0> / <shop_1_0> / <shop_2_0>
// 的 token ID，在模型加载时通过 initFromCkptPath() 一次性解析。
//
// 当前硬编码 shop_0/1/2 三层，后续可扩展为可配置。
// ---------------------------------------------------------------------------
class CsrLayerBaseIds {
public:
    static CsrLayerBaseIds& instance() {
        static CsrLayerBaseIds inst;
        return inst;
    }

    // 从 ckpt_path/tokenizer_config.json 的 added_tokens_decoder 中查找
    // <shop_0_0>, <shop_1_0>, <shop_2_0> 的 token ID。
    // 如果文件不存在或 token 未找到，保持 initialized_ = false，decode 回退到直接使用 uint16 值。
    void initFromCkptPath(const std::string& ckpt_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_ = false;

        // 目标 token 名称（硬编码三层）
        const std::string target_tokens[3] = {"<shop_0_0>", "<shop_1_0>", "<shop_2_0>"};
        bool found[3] = {false, false, false};

        // 读取 tokenizer_config.json
        std::string config_path = ckpt_path + "/tokenizer_config.json";
        std::ifstream file(config_path);
        if (!file.is_open()) {
            // tokenizer_config.json 不存在，尝试 tokenizer 子目录
            config_path = ckpt_path + "/tokenizer/tokenizer_config.json";
            file.open(config_path);
            if (!file.is_open()) {
                return;  // 文件不存在，保持未初始化
            }
        }

        std::ostringstream ss;
        ss << file.rdbuf();
        const std::string content = ss.str();

        // 逐个查找目标 token：
        // added_tokens_decoder 格式: "TOKEN_ID": {"content": "<shop_0_0>", ...}
        // 查找 "content": "<shop_X_0>" 后向前回溯找到对应的 key（token ID）
        for (int layer = 0; layer < 3; ++layer) {
            std::string pattern = "\"content\": \"" + target_tokens[layer] + "\"";
            size_t pos = content.find(pattern);
            if (pos == std::string::npos) {
                // 尝试无空格格式
                pattern = "\"content\":\"" + target_tokens[layer] + "\"";
                pos = content.find(pattern);
            }
            if (pos == std::string::npos) continue;

            // 向前查找最近的 "DIGITS": { 模式（即 added_tokens_decoder 的 key）
            // 从 pos 向前搜索 '{'，再向前找 '"DIGITS"'
            size_t brace = content.rfind('{', pos);
            if (brace == std::string::npos || brace == 0) continue;

            // 在 '{' 前面找到类似 "152001": 的模式
            size_t colon = content.rfind(':', brace - 1);
            if (colon == std::string::npos) continue;

            // 提取 colon 左边的 quoted string
            size_t quote_end = content.rfind('"', colon - 1);
            if (quote_end == std::string::npos) continue;
            size_t quote_start = content.rfind('"', quote_end - 1);
            if (quote_start == std::string::npos) continue;

            std::string key_str = content.substr(quote_start + 1, quote_end - quote_start - 1);
            try {
                base_ids_[layer] = std::stoi(key_str);
                found[layer] = true;
            } catch (...) {
                continue;
            }
        }

        if (found[0] && found[1] && found[2]) {
            initialized_ = true;
        }
    }

    bool initialized() const { return initialized_; }

    // 获取第 layer 层的 base token ID（0-indexed）
    int baseId(int layer) const { return base_ids_[layer]; }

    // 获取所有 base IDs 的指针（用于批量操作）
    const int* baseIds() const { return base_ids_; }

private:
    CsrLayerBaseIds() = default;
    CsrLayerBaseIds(const CsrLayerBaseIds&) = delete;
    CsrLayerBaseIds& operator=(const CsrLayerBaseIds&) = delete;

    std::mutex mutex_;
    bool initialized_ = false;
    int  base_ids_[3] = {0, 0, 0};
};

// sids<N>：N个token ID组成的定长数组，表示一条约束路径
template<int N>
struct sids {
    int rq_id[N];
    uint64_t packed_key = 0;  // 预计算的 64-bit 打包 key，用于 N==3 时加速排序

    __attribute__((always_inline)) bool operator<(const sids<N>& other) const {
        if constexpr (N == 3) {
            // N==3 时使用预计算的 packed_key，单次 64-bit 整数比较
            return packed_key < other.packed_key;
        } else {
            // 显式展开前几个比较，消除小 N 时的循环开销，确保编译器内联最优
            if (rq_id[0] != other.rq_id[0]) return rq_id[0] < other.rq_id[0];
            if (N > 1 && rq_id[1] != other.rq_id[1]) return rq_id[1] < other.rq_id[1];
            if (N > 2 && rq_id[2] != other.rq_id[2]) return rq_id[2] < other.rq_id[2];
            for (int i = 3; i < N - 1; ++i) {
                if (rq_id[i] != other.rq_id[i]) return rq_id[i] < other.rq_id[i];
            }
            return rq_id[N - 1] < other.rq_id[N - 1];
        }
    }
};

// CSRIndex<N>：对长度为N的约束路径集合构建的CSR格式前缀树索引。
// 每次请求在CPU上构建一次，然后以flat int32数组形式上传到GPU。
//
// 数据布局（对应 csr_utils.py 中的 build_static_index）：
//   packed_csr_tokens/states : [边数 + vocab_size]  (token, next_state) 转移对
//   indptr                   : [状态数 + 2]  行指针数组
//   start_mask               : [vocab_size]  第0层（根节点出发）的合法token掩码
//   layer_max_branches       : [N]  每层前缀树的最大分支数
template<int N>
struct CSRIndex {
    std::vector<int>  indptr;             // 行指针数组，大小 = 状态数 + 2
    std::vector<int>  packed_csr_tokens;  // 转移表的token列
    std::vector<int>  packed_csr_states;  // 转移表的next_state列
    std::vector<bool> start_mask;         // 第0层合法token掩码，大小 = vocab_size
    std::vector<int>  layer_max_branches; // 每层最大分支数，大小 = N

    int num_states = 0;
    int vocab_size = 0;

    CSRIndex() = default;
};

// ---------------------------------------------------------------------------
// build_csr_from_fresh_data<N>
//
// 从已排序的 sids<N> 数组构建 CSRIndex<N>。
// 对应 csr_utils.py 中的 build_static_index（仅 CSR 路径，无 dense head）。
//
// 状态ID分配规则：
//   state 0      ：根节点（隐式，CSR中无对应行，只有末尾填充行）
//   state 1..V   ：第0层前缀树节点（第一个token t -> state t+1）
//   state V+1..  ：更深层节点，按 DFS 顺序递增分配
//
// packed_csr 末尾追加 vocab_size 个填充行（token=V, next_state=0），
// 保证越界索引安全（与 Python 版本一致）。
// ---------------------------------------------------------------------------
template<int N>
bool build_csr_from_fresh_data(const std::vector<sids<N>>& fresh_data,
                                CSRIndex<N>&                index,
                                int                         vocab_size) {
    if (fresh_data.empty()) {
        return false;
    }

    const int data_size = static_cast<int>(fresh_data.size());

    index.vocab_size = vocab_size;

    // --- 1. 构建第0层合法token掩码（start_mask） ---
    index.start_mask.assign(vocab_size, false);
    int depth0_count = 0;
    for (const auto& s : fresh_data) {
        int t0 = s.rq_id[0];
        if (t0 >= 0 && t0 < vocab_size && !index.start_mask[t0]) {
            index.start_mask[t0] = true;
            ++depth0_count;
        }
    }

    // --- 2. is_new[i][j]：第i行是否在第j层引入了新的前缀树节点 ---
    // 扁平化为一维连续数组，提升缓存局部性
    std::vector<char> is_new(data_size * N, 0);
    for (int j = 0; j < N; ++j) {
        is_new[j] = 1;
    }
    for (int i = 1; i < data_size; ++i) {
        is_new[i * N + 0] = (fresh_data[i].rq_id[0] != fresh_data[i - 1].rq_id[0]);
        for (int j = 1; j < N; ++j) {
            if (!is_new[i * N + j - 1] && fresh_data[i].rq_id[j] == fresh_data[i - 1].rq_id[j]) {
                is_new[i * N + j] = 0;
            } else {
                is_new[i * N + j] = 1;
            }
        }
    }

    // --- 3. 状态ID分配 ---
    // 第0层节点：state_id = token_id + 1（占用 state 1..vocab_size）
    // 更深层节点：从 vocab_size+1 开始顺序分配
    std::vector<int> state_ids(data_size * (N - 1), 0);
    for (int i = 0; i < data_size; ++i) {
        state_ids[i * (N - 1) + 0] = fresh_data[i].rq_id[0] + 1;
    }

    int num_states = vocab_size;  // 下一个可用状态ID - 1
    for (int depth = 1; depth < N - 1; ++depth) {
        for (int i = 0; i < data_size; ++i) {
            if (i == 0 || is_new[i * N + depth]) {
                ++num_states;
                state_ids[i * (N - 1) + depth] = num_states;
            } else {
                state_ids[i * (N - 1) + depth] = state_ids[(i - 1) * (N - 1) + depth];
            }
        }
    }
    ++num_states;  // 最终状态总数 = 已分配的最大状态ID
    index.num_states = num_states;

    // --- 4. 收集前缀树的所有边 ---
    std::vector<int> parent_ids_vec;
    std::vector<int> token_ids_vec;
    std::vector<int> child_ids_vec;
    const int max_edges = data_size * (N - 1);
    parent_ids_vec.reserve(max_edges);
    token_ids_vec.reserve(max_edges);
    child_ids_vec.reserve(max_edges);

    // layer_max_branches[0] = 第0层不同起始token的个数
    index.layer_max_branches.clear();
    index.layer_max_branches.push_back(depth0_count);

    for (int depth = 1; depth < N; ++depth) {
        int max_branches = 0;
        int cur_branches = 0;
        int prev_parent  = -1;
        for (int i = 0; i < data_size; ++i) {
            if (is_new[i * N + depth]) {
                int parent = state_ids[i * (N - 1) + depth - 1];
                int token  = fresh_data[i].rq_id[depth];
                int child  = (depth < N - 1) ? state_ids[i * (N - 1) + depth] : 0;  // 0 表示终止状态
                parent_ids_vec.emplace_back(parent);
                token_ids_vec.emplace_back(token);
                child_ids_vec.emplace_back(child);

                if (parent != prev_parent) {
                    cur_branches = 1;
                    prev_parent = parent;
                } else {
                    ++cur_branches;
                }
                if (cur_branches > max_branches) max_branches = cur_branches;
            }
        }
        index.layer_max_branches.push_back(max_branches);
    }

    // --- 5. 构建 indptr ---
    // indptr 大小 = num_states + 2：
    //   第 0..num_states-1 行对应 state 0..num_states-1
    //   末尾多一行填充（state num_states），用于越界安全索引
    const int indptr_size = num_states + 2;
    index.indptr.assign(indptr_size, 0);
    for (int pid : parent_ids_vec) {
        index.indptr[pid + 1]++;
    }
    for (int i = 1; i < indptr_size - 1; ++i) {
        index.indptr[i] += index.indptr[i - 1];
    }
    // 末尾填充行：占用 vocab_size 个槽位，用于越界安全
    index.indptr[num_states + 1] = index.indptr[num_states] + vocab_size;

    // --- 6. 构建 packed_csr（转移表） ---
    const int num_transitions = static_cast<int>(token_ids_vec.size());
    const int packed_size     = num_transitions + vocab_size;
    index.packed_csr_tokens.resize(packed_size);
    index.packed_csr_states.resize(packed_size);

    for (int i = 0; i < num_transitions; ++i) {
        index.packed_csr_tokens[i] = token_ids_vec[i];
        index.packed_csr_states[i] = child_ids_vec[i];
    }
    // 填充行：token = vocab_size（哨兵值），state = 0
    for (int i = 0; i < vocab_size; ++i) {
        index.packed_csr_tokens[num_transitions + i] = vocab_size;
        index.packed_csr_states[num_transitions + i] = 0;
    }

    return true;
}

// ---------------------------------------------------------------------------
// parse_sid_from_string_view<N>
// 单次遍历解析 "t0_t1_t2" 格式，无需 find('_')，零拷贝。
// ---------------------------------------------------------------------------
template<int N>
inline sids<N> parse_sid_from_string_view(std::string_view sv) {
    sids<N> sid{};
    const char* p   = sv.data();
    const char* end = p + sv.size();
    for (int j = 0; j < N && p < end; ++j) {
        int value = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            value = value * 10 + (*p - '0');
            ++p;
        }
        sid.rq_id[j] = value;
        if (p < end && *p == '_') {
            ++p;
        }
    }
    if constexpr (N == 3) {
        // 每个字段分配 18 bits（最大支持 262143），覆盖千问3 词表（~152064）
        sid.packed_key = (uint64_t(sid.rq_id[0]) << 36) |
                         (uint64_t(sid.rq_id[1]) << 18) |
                         uint64_t(sid.rq_id[2]);
    }
    return sid;
}

// ---------------------------------------------------------------------------
// split_strings<N>
// 将 "t0_t1_t2" 格式的字符串解析为 sids<N>。
// ---------------------------------------------------------------------------
template<int N>
std::vector<sids<N>> split_strings(const std::vector<std::string>& ele_rq_ids) {
    std::vector<sids<N>> result;
    result.reserve(ele_rq_ids.size());

    for (const std::string& str : ele_rq_ids) {
        result.emplace_back(parse_sid_from_string_view<N>(std::string_view(str)));
    }
    return result;
}

// ---------------------------------------------------------------------------
// parseJsonArray<N>
// 从 JSON 文件中解析 [[t0,t1,...], ...] 格式的数组，返回 sids<N> 列表。
// ---------------------------------------------------------------------------
template<int N>
std::vector<sids<N>> parseJsonArray(const std::string& filename) {
    std::vector<sids<N>> result;
    std::ifstream        file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return result;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    content.erase(std::remove_if(content.begin(), content.end(), ::isspace), content.end());

    size_t outer_start = content.find('[');
    size_t outer_end   = content.rfind(']');
    if (outer_start == std::string::npos || outer_end == std::string::npos) {
        std::cerr << "错误：JSON格式非法" << std::endl;
        return result;
    }

    size_t pos = outer_start + 1;
    while (pos < outer_end) {
        size_t array_start = content.find('[', pos);
        if (array_start == std::string::npos || array_start >= outer_end) break;
        size_t array_end = content.find(']', array_start);
        if (array_end == std::string::npos) break;

        std::string        inner = content.substr(array_start + 1, array_end - array_start - 1);
        std::stringstream  ss(inner);
        std::string        token;
        std::vector<int>   row;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) row.push_back(std::stoi(token));
        }

        if (static_cast<int>(row.size()) == N) {
            sids<N> s{};
            for (int i = 0; i < N; ++i) s.rq_id[i] = row[i];
            result.push_back(s);
        }
        pos = array_end + 1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// decodePackedEleRqIds<N>
//
// 从 base64 编码的 packed uint64 二进制数据解码为 sids<N>。
//
// 二进制格式：
//   [uint32_t count]            — packed ID 个数
//   [uint64_t packed_ids[count]] — 每个 packed_id = (t0 << 36) | (t1 << 18) | t2
//
// base64 编码后嵌入 JSON 的 ele_rq_ids_pb 字段，替代 50000 个字符串的 JSON 数组，
// 将 JSON 解析耗时从 ~25ms 降至 ~5ms（1 个字符串分配 vs 50000 个）。
// ---------------------------------------------------------------------------
template<int N>
std::vector<sids<N>> decodePackedEleRqIds(const std::string& base64_str) {
    std::vector<sids<N>> result;
    if (base64_str.empty()) {
        return result;
    }

    // base64 解码 → 原始字节
    std::string raw = autil::legacy::Base64DecodeFast(base64_str);
    if (raw.size() < sizeof(uint32_t)) {
        return result;
    }

    // 读取 count
    uint32_t count = 0;
    std::memcpy(&count, raw.data(), sizeof(uint32_t));

    // 计算实际可用的 packed ID 个数（数据不足的尾部忽略）
    const size_t payload_bytes = raw.size() - sizeof(uint32_t);
    const uint32_t available_count = static_cast<uint32_t>(payload_bytes / sizeof(uint64_t));
    const uint32_t actual_count = std::min(count, available_count);
    if (actual_count == 0) {
        return result;
    }

    result.reserve(actual_count);
    const uint64_t* packed = reinterpret_cast<const uint64_t*>(raw.data() + sizeof(uint32_t));

    for (uint32_t i = 0; i < actual_count; ++i) {
        sids<N> sid{};
        uint64_t pk = packed[i];
        if constexpr (N == 3) {
            // 每个字段 18 bits（最大 262143），与 packed_key 编码一致
            sid.rq_id[0] = static_cast<int>((pk >> 36) & 0x3FFFF);
            sid.rq_id[1] = static_cast<int>((pk >> 18) & 0x3FFFF);
            sid.rq_id[2] = static_cast<int>(pk & 0x3FFFF);
            sid.packed_key = pk;
        } else {
            // 通用回退：按 21 bits 均分（3 × 21 = 63 bits）
            for (int j = 0; j < N && j < 3; ++j) {
                sid.rq_id[j] = static_cast<int>((pk >> (42 - j * 21)) & 0x1FFFFF);
            }
            if constexpr (N == 3) {
                sid.packed_key = pk;
            }
        }
        result.emplace_back(sid);
    }

    return result;
}

// ---------------------------------------------------------------------------
// decodeFlatUint16EleRqIds<N>
//
// 从 base64 编码的 flat uint16 二进制数据解码为 sids<N>。
// 每组 3 个 uint16 直接存储 token 偏移量，无需位运算拆包。
//
// 二进制格式（无 header，纯 uint16 数组）：
//   [uint16_t data[triple_count*3]]  — 每 3 个连续 uint16 为一组 (t0, t1, t2)
//   triple_count = total_bytes / (sizeof(uint16_t) * 3)
//   不足 3 个 uint16 的尾部忽略
//
// 注意：uint16 最大 65535，仅覆盖词表 ≤65535 的模型。
// ---------------------------------------------------------------------------
template<int N>
std::vector<sids<N>> decodeFlatUint16EleRqIds(const std::string& base64_str) {
    std::vector<sids<N>> result;
    if (base64_str.empty()) {
        return result;
    }

    // base64 解码 → 原始字节
    std::string raw = autil::legacy::Base64DecodeFast(base64_str);

    // 无 header：直接从总字节数计算完整三元组个数
    const size_t bytes_per_triple = sizeof(uint16_t) * 3;  // 6 bytes
    const uint32_t actual_count = static_cast<uint32_t>(raw.size() / bytes_per_triple);
    if (actual_count == 0) {
        return result;
    }

    result.reserve(actual_count);
    const uint16_t* data = reinterpret_cast<const uint16_t*>(raw.data());

    // extra_info 中的 uint16 值是偏移量，需要加上每层的 base token ID 才是实际 token ID。
    // CsrLayerBaseIds 在模型加载时从 tokenizer_config.json 初始化。
    // 未初始化时回退到直接使用 uint16 值（向后兼容）。
    const auto& bases = CsrLayerBaseIds::instance();
    const bool apply_offset = bases.initialized();
    const int base0 = apply_offset ? bases.baseId(0) : 0;
    const int base1 = apply_offset ? bases.baseId(1) : 0;
    const int base2 = apply_offset ? bases.baseId(2) : 0;

    for (uint32_t i = 0; i < actual_count; ++i) {
        sids<N> sid{};
        if constexpr (N == 3) {
            sid.rq_id[0] = base0 + static_cast<int>(data[i * 3 + 0]);
            sid.rq_id[1] = base1 + static_cast<int>(data[i * 3 + 1]);
            sid.rq_id[2] = base2 + static_cast<int>(data[i * 3 + 2]);
            // 重新计算 packed_key 以保证排序正确
            sid.packed_key = (uint64_t(sid.rq_id[0]) << 36) |
                             (uint64_t(sid.rq_id[1]) << 18) |
                             uint64_t(sid.rq_id[2]);
        } else {
            const int bases_arr[3] = {base0, base1, base2};
            for (int j = 0; j < N && j < 3; ++j) {
                sid.rq_id[j] = bases_arr[j] + static_cast<int>(data[i * 3 + j]);
            }
            if constexpr (N == 3) {
                sid.packed_key = (uint64_t(sid.rq_id[0]) << 36) |
                                 (uint64_t(sid.rq_id[1]) << 18) |
                                 uint64_t(sid.rq_id[2]);
            }
        }
        result.emplace_back(sid);
    }

    return result;
}