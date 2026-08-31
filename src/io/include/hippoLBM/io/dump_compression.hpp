/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */

#pragma once

#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace hippoLBM {

constexpr int DUMP_COMPRESSION_FASTEST = Z_BEST_SPEED;
constexpr int DUMP_COMPRESSION_DEFAULT = Z_DEFAULT_COMPRESSION;
constexpr int DUMP_COMPRESSION_BEST = Z_BEST_COMPRESSION;

/** @brief Compresses a raw buffer with zlib. Falls back to storing it raw if compression doesn't shrink it. */
inline std::vector<uint8_t> zlib_compress(const void* src, uint64_t src_bytes, int compression_level,
                                          bool& is_compressed) {
  uLongf bound = compressBound(static_cast<uLong>(src_bytes));
  std::vector<uint8_t> dst(bound);
  uLongf dst_len = bound;
  int ret = compress2(reinterpret_cast<Bytef*>(dst.data()), &dst_len, reinterpret_cast<const Bytef*>(src),
                      static_cast<uLong>(src_bytes), compression_level);
  if (ret != Z_OK) {
    std::cerr << "[dump_compression] zlib compression failed (error code " << ret << ")." << std::endl;
    dst_len = static_cast<uLongf>(src_bytes) + 1;  // force the raw fallback below
  }
  if (dst_len >= src_bytes) {
    dst.resize(src_bytes);
    std::memcpy(dst.data(), src, src_bytes);
    is_compressed = false;
  } else {
    dst.resize(dst_len);
    is_compressed = true;
  }
  return dst;
}

/** @brief Decompresses a buffer previously produced by zlib_compress() with is_compressed == true. */
inline std::vector<uint8_t> zlib_decompress(const void* src, uint64_t compressed_size, uint64_t uncompressed_size) {
  std::vector<uint8_t> dst(uncompressed_size);
  uLongf dst_len = static_cast<uLongf>(uncompressed_size);
  int ret = uncompress(reinterpret_cast<Bytef*>(dst.data()), &dst_len, reinterpret_cast<const Bytef*>(src),
                       static_cast<uLong>(compressed_size));
  if (ret != Z_OK || dst_len != uncompressed_size) {
    std::cerr << "[dump_compression] zlib decompression failed (error code " << ret << ")." << std::endl;
  }
  return dst;
}

}  // namespace hippoLBM
