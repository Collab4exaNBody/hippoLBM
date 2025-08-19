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

namespace hippoLBM
{
	enum Area
	{
		Local,  /**< Refers to a local region or domain */
		Global  /**< Refers to the entire system or global domain */
	};

	enum Side
	{
		Left,   /**< Refers to the left side */
		Right   /**< Refers to the right side */
	};

	enum Traversal
	{
		All,         ///< All points in the grid
		Real,        ///< All points excluding the ghost layer
		Inside,      ///< All points excluding the ghost layer and an additional layer of size 1
		Edge,        ///< Boundary points only (excluding Inside)
		Ghost_Edge,  ///< Boundary points including the ghost layer
		Plan_xy_0,   ///< Plane at z = 0 in the XY direction
		Plan_xy_l,   ///< Plane at z = L in the XY direction
		Plan_xz_0,   ///< Plane at y = 0 in the XZ direction
		Plan_xz_l,   ///< Plane at y = L in the XZ direction
		Plan_yz_0,   ///< Plane at x = 0 in the YZ direction
		Plan_yz_l,   ///< Plane at x = L in the YZ direction
		Extend       ///< Used for ParaView output or to test if the grid contains a point
	};

	/**
	 * @brief Dimension identifiers used for indexing in 3D space.
	 */
	constexpr int DIMX = 0; ///< X dimension index
	constexpr int DIMY = 1; ///< Y dimension index
	constexpr int DIMZ = 2; ///< Z dimension index
	constexpr int DIM_MAX = 3; ///< Total number of spatial dimensions
}

#define FLUIDE_ -1
#define WALL_ -2
