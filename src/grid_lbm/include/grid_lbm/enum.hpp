#pragma once

namespace hipoLBM
{
	enum Area
	{
		Local,
		Global
	};

	enum Traversal
	{
		All, ///< All points into a grid
		Real, ///< All points - ghost layer
		Inside, ///< All points - ghost layer - 1 layer of size 1
		Edge, ///< Read whithout Inside
		Ghost_Edge, ///< All without Inside
		Plan_xy_0,
		Plan_xy_l,
		Extend ///< used for paraview and test if the grid have a point
	};
}
