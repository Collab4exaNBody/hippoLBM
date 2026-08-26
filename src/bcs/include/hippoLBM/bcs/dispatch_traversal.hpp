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

// onika
#include <onika/log.h>
#include <onika/parallel/parallel_for.h>

// hippoLBM
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <string>
#include <utility>

namespace hippoLBM {
namespace bcs {

/**
 * @brief Dispatches a runtime Traversal value.
 */
template <template <int, Traversal> class Functor, int Q, class PtrT, class CtxFunc, class... Args>
inline void dispatch_traversal(CtxFunc&& make_ctx, Traversal t, const std::string& kernel_name, PtrT ptr, size_t size,
                               Args&&... args) {
  switch (t) {
    case Traversal::Plan_yz_0: {
      Functor<Q, Traversal::Plan_yz_0> bc = {};
      parallel_for_id(ptr, size, bc, make_ctx(kernel_name.c_str()), std::forward<Args>(args)...);
      break;
    }
    case Traversal::Plan_yz_l: {
      Functor<Q, Traversal::Plan_yz_l> bc = {};
      parallel_for_id(ptr, size, bc, make_ctx(kernel_name.c_str()), std::forward<Args>(args)...);
      break;
    }
    case Traversal::Plan_xz_0: {
      Functor<Q, Traversal::Plan_xz_0> bc = {};
      parallel_for_id(ptr, size, bc, make_ctx(kernel_name.c_str()), std::forward<Args>(args)...);
      break;
    }
    case Traversal::Plan_xz_l: {
      Functor<Q, Traversal::Plan_xz_l> bc = {};
      parallel_for_id(ptr, size, bc, make_ctx(kernel_name.c_str()), std::forward<Args>(args)...);
      break;
    }
    case Traversal::Plan_xy_0: {
      Functor<Q, Traversal::Plan_xy_0> bc = {};
      parallel_for_id(ptr, size, bc, make_ctx(kernel_name.c_str()), std::forward<Args>(args)...);
      break;
    }
    case Traversal::Plan_xy_l: {
      Functor<Q, Traversal::Plan_xy_l> bc = {};
      parallel_for_id(ptr, size, bc, make_ctx(kernel_name.c_str()), std::forward<Args>(args)...);
      break;
    }
    default:
      lout << "[bcs] dispatch_traversal: ignoring unknown region" << std::endl;
      break;
  }
}

}  // namespace bcs
}  // namespace hippoLBM
