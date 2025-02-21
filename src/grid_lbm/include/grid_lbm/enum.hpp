#pragma once

namespace hippoLBM
{
	enum Area
	{
		Local,
		Global
	};

	enum Direction {Left, Right};

	enum Traversal
	{
		All, ///< All points into a grid
		Real, ///< All points - ghost layer
		Inside, ///< All points - ghost layer - 1 layer of size 1
		Edge, ///< Read whithout Inside
		Ghost_Edge, ///< All without Inside
		Plan_xy_0,
		Plan_xy_l,
		Plan_xz_0,
		Plan_xz_l,
		Plan_yz_0,
		Plan_yz_l,
		Extend ///< used for paraview and test if the grid have a point
	};
}
