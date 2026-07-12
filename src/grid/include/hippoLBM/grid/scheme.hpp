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

namespace hippoLBM {
// number of unknowns fi in 3DQ19
template <int Q>
struct LBMScheme {};

template <int EX, int EY, int EZ, int WDen, int IOPP>
struct Direction {
  static constexpr int ex = EX;
  static constexpr int ey = EY;
  static constexpr int ez = EZ;
  static constexpr double w = 1.0 / WDen;
  static constexpr int iopp = IOPP;
};

template <typename... Dirs>
struct Scheme {
  static constexpr int Q = sizeof...(Dirs);

  template <int I>
  using dir = std::tuple_element_t<I, std::tuple<Dirs...>>;

  template <int I>
  static constexpr int iopp_of = dir<I>::iopp;
};

template <>
struct LBMScheme<19> {
  template <typename T>
  using vector_t = onika::memory::CudaMMVector<T>;
  // Ex Ey Ez 1/W IOPP
  using Coefficients = Scheme<Direction<0, 0, 0, 3, 0>,      // 0  -> 0
                              Direction<1, 0, 0, 18, 2>,     // 1  -> 2
                              Direction<-1, 0, 0, 18, 1>,    // 2  -> 1
                              Direction<0, 1, 0, 18, 4>,     // 3  -> 4
                              Direction<0, -1, 0, 18, 3>,    // 4  -> 3
                              Direction<0, 0, 1, 18, 6>,     // 5  -> 6
                              Direction<0, 0, -1, 18, 5>,    // 6  -> 5
                              Direction<1, 1, 0, 36, 8>,     // 7  -> 8
                              Direction<-1, -1, 0, 36, 7>,   // 8  -> 7
                              Direction<1, -1, 0, 36, 10>,   // 9  -> 10
                              Direction<-1, 1, 0, 36, 9>,    // 10 -> 9
                              Direction<1, 0, 1, 36, 12>,    // 11 -> 12
                              Direction<-1, 0, -1, 36, 11>,  // 12 -> 11
                              Direction<1, 0, -1, 36, 14>,   // 13 -> 14
                              Direction<-1, 0, 1, 36, 13>,   // 14 -> 13
                              Direction<0, 1, 1, 36, 16>,    // 15 -> 16
                              Direction<0, -1, -1, 36, 15>,  // 16 -> 15
                              Direction<0, 1, -1, 36, 18>,   // 17 -> 18
                              Direction<0, -1, 1, 36, 17>>;  // 18 -> 17
  constexpr static int Q = Coefficients::Q;
  static_assert(Q == 19, "LBMScheme<19> should have 19 directions");
};
}  // namespace hippoLBM
