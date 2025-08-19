#pragma once

#include <onika/math/basic_types_def.h>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/point3d.hpp>
#include <hippoLBM/grid/box3d.hpp>
#include <cassert>
#include <tuple>

namespace hippoLBM
{
  /**
   * @brief Lattice Boltzmann Method grid representation.
   * 
   * Stores the local grid, extended grid, offset, ghost layers, and spatial resolution.
   */
  struct LBMGrid
  {
    static constexpr int DIM = 3; ///< Number of spatial dimensions

    Box3D bx;        ///< Box covering the actual computational grid
    Box3D ext;       ///< Box covering the extended grid (includes ghost layers)
    Point3D offset;  ///< Offset of the real grid (useful for global indexing)
    int ghost_layer = 2; ///< Number of ghost layers (default 2 for DEM + LBM)
    double dx;       ///< Distance between two grid points (spatial resolution)

    LBMGrid() {};

    LBMGrid(Box3D& b, Point3D& o, const int g, double d) : bx(b), offset(o), ghost_layer(g), dx(d)
    {
      // add a print function here ?
    }

    // setter
    inline void set_box(Box3D& b) { bx = b;}
    inline void set_ext(Box3D& e) {ext = e;}
    inline void set_offset(Point3D& o) {offset = o;}
    inline void set_offset(Point3D&& o) {offset = o;}
    inline void set_ghost_layer(const int g) {ghost_layer = g;}
    inline void set_dx(const double d) {dx = d;}

    /**
     * @brief Computes the starting index of a given dimension for a subdomain traversal.
     *
     * Depending on the traversal mode (`Traversal`) and the area (`Area`), 
     * this function returns the correct lower bound index in the given dimension.
     *
     * @tparam A  Specifies whether the index is expressed in the local or global coordinate system.
     * @tparam Tr Traversal type (All, Real, Inside, or Extend).
     *
     * @param dim Dimension index (DIMX, DIMY, or DIMZ).
     * @return Starting index along the given dimension.
     *
     * @note
     * - `Traversal::All` → Full box range.  
     * - `Traversal::Real` → Excludes ghost layer.  
     * - `Traversal::Inside` → Excludes ghost layer + one additional layer.  
     * - `Traversal::Extend` → Uses the extended box.  
     * - If `Area::Global`, the global offset is added.  
     */
    template<Area A, Traversal Tr>
      ONIKA_HOST_DEVICE_FUNC inline int start(const int dim) const
      {
        //static_assert(A == Area::Local);
        int res = 0;
        static_assert(Tr == Traversal::All || Tr == Traversal::Real || Tr == Traversal::Inside || Tr == Traversal::Extend);
        if constexpr( Tr == Traversal::All ) res = bx.start(dim);
        if constexpr( Tr == Traversal::Real ) res = bx.start(dim) + ghost_layer;
        if constexpr( Tr == Traversal::Inside ) res = bx.start(dim) + ghost_layer + 1;
        if constexpr( Tr == Traversal::Extend ) res = ext.start(dim);
        if constexpr( A == Area::Global ) res += offset[dim];
        return res;
      }


    /**
     * @brief Computes the ending index of a given dimension for a subdomain traversal.
     *
     * Depending on the traversal mode (`Traversal`) and the area (`Area`), 
     * this function returns the correct upper bound index in the given dimension.
     *
     * @tparam A  Specifies whether the index is expressed in the local or global coordinate system.
     * @tparam Tr Traversal type (All, Real, Inside, or Extend).
     *
     * @param dim Dimension index (DIMX, DIMY, or DIMZ).
     * @return Ending index along the given dimension (exclusive).
     *
     * @note
     * - `Traversal::All` → Full box range.  
     * - `Traversal::Real` → Excludes ghost layer.  
     * - `Traversal::Inside` → Excludes ghost layer + one additional layer.  
     * - `Traversal::Extend` → Uses the extended box.  
     * - If `Area::Global`, the global offset is added.  
     */
    template<Area A, Traversal Tr>
      ONIKA_HOST_DEVICE_FUNC inline int end(const int dim) const
      {
        //static_assert(A == Area::Local);
        static_assert(Tr == Traversal::All || Tr == Traversal::Real || Tr == Traversal::Inside ||  Tr == Traversal::Extend );

        int res = 0;
        if constexpr ( Tr == Traversal::All ) res = bx.end(dim);
        if constexpr ( Tr == Traversal::Real ) res = bx.end(dim) - ghost_layer;
        if constexpr ( Tr == Traversal::Inside ) res = bx.end(dim) - ghost_layer - 1;
        if constexpr ( Tr == Traversal::Extend ) res = ext.end(dim);
        if constexpr ( A == Area::Global ) res += offset[dim];
        return res;
      }


    /**
     * @brief Builds a 3D bounding box for the specified traversal and area.
     *
     * Depending on the traversal mode (`Traversal`) and the area (`Area`), 
     * this function constructs the corresponding 3D box.
     *
     * @tparam A  Specifies whether the box is defined in local or global coordinates.
     * @tparam Tr Traversal type (All, Extend, Real, Inside).
     *
     * @return A Box3D object representing the requested region.
     *
     * @note
     * - `(Area::Local, Traversal::All)` returns the full local box (`bx`).  
     * - `(Area::Local, Traversal::Extend)` returns the extended box (`ext`).  
     * - For other cases, the box is built from `start<A,Tr>(dim)` and `end<A,Tr>(dim)` for each dimension.  
     */
    template<Area A, Traversal Tr>
      ONIKA_HOST_DEVICE_FUNC Box3D build_box() const
      {
        if constexpr (A == Area::Local && Tr == Traversal::All) return bx;
        if constexpr (A == Area::Local && Tr == Traversal::Extend) return ext;
        Point3D lower, upper;
        for(int dim = 0; dim < DIM ; dim++)
        {
          lower[dim] = this->start<A,Tr>(dim);
          upper[dim] = this->end<A,Tr>(dim);
        }
        Box3D res = {lower, upper};
        return res;
      }

    /**
     * @brief Checks whether a point lies inside the specified region of the grid.
     *
     * The test is performed according to the given traversal mode (`Traversal`) 
     * and area (`Area`).
     *
     * @tparam A  Specifies whether the region is expressed in local or global coordinates.
     * @tparam Tr Traversal type (All, Real, Inside, Extend).
     *
     * @param p Point coordinates to be tested.
     * @return true if the point lies inside the region, false otherwise.
     *
     * @note
     * - The lower bound (`start`) is inclusive.  
     * - The upper bound (`end`) is exclusive in typical loop usage, but here 
     *   the check uses `<= end`, meaning the boundary point is considered **inside**.  
     */
    template<Area A, Traversal Tr>
      ONIKA_HOST_DEVICE_FUNC inline bool contains(Point3D& p) const
      {
        for (int dim = 0; dim < DIM ; dim++)
        {
          if(p[dim] < this->start<A,Tr>(dim) || p[dim] > this->end<A,Tr>(dim))
          {
            return false;
          }
        }
        return true;
      }

    /**
     * @brief Checks whether a point lies inside the extended grid region.
     *
     * A point is considered defined if all its coordinates are within the 
     * lower and upper bounds of the extended box (`ext`).
     *
     * @param p Point coordinates to be tested.
     * @return true if the point is inside the extended region, false otherwise.
     *
     * @note
     * - The lower bound (`ext.start(dim)`) is inclusive.  
     * - The upper bound (`ext.end(dim)`) is also inclusive.  
     */
    ONIKA_HOST_DEVICE_FUNC inline bool is_defined(Point3D& p) const
    {
      for (int dim = 0; dim < DIM ; dim++)
      {
        if(p[dim] < ext.start(dim) || p[dim] > ext.end(dim))
        {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Checks whether a given 3D index lies inside the extended grid region.
     *
     * Equivalent to `is_defined(Point3D)` but takes explicit integer 
     * indices for convenience.
     *
     * @param i Index along X dimension.
     * @param j Index along Y dimension.
     * @param k Index along Z dimension.
     * @return true if the indices are inside the extended region, false otherwise.
     *
     * @note
     * - The lower bound (`ext.start(dim)`) is inclusive.  
     * - The upper bound (`ext.end(dim)`) is also inclusive.  
     */
    ONIKA_HOST_DEVICE_FUNC inline bool is_defined(int i, int j, int k) const
    {
      if( i < ext.start(0) || i > ext.end(0) ) return false;
      if( j < ext.start(1) || j > ext.end(1) ) return false;
      if( k < ext.start(2) || k > ext.end(2) ) return false;
      return true;
    }

    /**
     * @brief Checks whether a point lies inside the local grid region.
     *
     * A point is considered local if all its coordinates fall within the 
     * bounds of the local box (`bx`).
     *
     * @param p Point coordinates to be tested.
     * @return true if the point is inside the local region, false otherwise.
     *
     * @note
     * - The lower bound (`bx.start(dim)`) is inclusive.  
     * - The upper bound (`bx.end(dim)`) is also inclusive.  
     */
    ONIKA_HOST_DEVICE_FUNC inline bool is_local(Point3D& p) const
    {
      for(int dim = 0 ; dim < DIM ; dim++)
      {
        if( p[dim] < bx.start(dim) || p[dim] > bx.end(dim) ) return false; 
      }
      return true;
    }

    /**
     * @brief Checks whether a point lies inside the global grid region.
     *
     * The global coordinates are first converted to local coordinates 
     * using the domain offset, then checked against the local box.
     *
     * @param p Point in global coordinates.
     * @return true if the point belongs to the global region, false otherwise.
     *
     * @note
     * - Internally, this computes `local = p + offset` and calls `is_local(local)`.  
     */
    ONIKA_HOST_DEVICE_FUNC inline bool is_global(Point3D& p) 
    {
      Point3D local = p + offset;
      return is_local(local);
    }

    /**
     * @brief Converts a point between local and global coordinate systems.
     *
     * This function converts a 3D point (x,y,z) either from local to global 
     * coordinates or from global to local coordinates depending on the 
     * template parameter `Area`.
     *
     * @tparam A     Conversion mode: `Area::Local` or `Area::Global`.
     * @tparam Check If true, performs an assertion to ensure the point lies 
     *               inside the valid grid region before conversion.
     *
     * @param x Coordinate along the X axis.
     * @param y Coordinate along the Y axis.
     * @param z Coordinate along the Z axis.
     *
     * @return A Point3D in the target coordinate system.
     *
     * @warning The `Check` option may incur a runtime cost due to the assertion. 
     * Use it mainly for debugging and validation.
     */
    template<Area A, bool Check=false>
      ONIKA_HOST_DEVICE_FUNC inline Point3D convert(int x, int y, int z) const
      {
        Point3D res = {x, y, z};
        static_assert ( A == Area::Local || A == Area::Global);
        if constexpr(A == Area::Local)
        {
          /** Convert global → local **/
          res = res - offset;
          /** Optional runtime check: point must be inside local domain **/
          if constexpr (Check)  assert(this->is_local(res));
        }

        if constexpr(A == Area::Global)
        {
          /** Optional runtime check: point must be valid before conversion **/
          if constexpr (Check)  assert(this->is_local(res));
          /** Convert local → global **/
          res = res + offset;
        }
        return res;
      }

    /**
     * @brief Projects a physical 3D point (Vec3d) onto the discrete grid.
     *
     * Converts a floating-point coordinate `r` into integer grid indices
     * and optionally converts between local/global coordinate systems.
     *
     * @tparam A     Conversion mode: `Area::Local` or `Area::Global`.
     * @tparam Check If true, asserts that the resulting point is within the grid.
     *
     * @param r A 3D vector representing the physical position.
     * @return Point3D corresponding to the discrete grid coordinates.
     *
     * @note
     * - Each coordinate is divided by `dx` and truncated to an integer.  
     * - The result is then passed to `convert<A,Check>` to handle the
     *   local/global conversion and optional validity check.
     */
    template<Area A, bool Check=false>
      ONIKA_HOST_DEVICE_FUNC inline Point3D project_to_grid(const onika::math::Vec3d&& r) const
      {
        Point3D p = {int(r.x / dx), int(r.y / dx), int(r.z / dx)};
        return convert<A,Check>(p[0], p[1], p[2]);
      }

    /*** convert a point to A area. */
    template<Area A, bool Check=false>
      ONIKA_HOST_DEVICE_FUNC inline Point3D convert(Point3D p) const
      {
        return convert<A,Check>(p[0], p[1], p[2]);
      }

    /*** @brief convert a point to A area. */
    template<Area A>
      ONIKA_HOST_DEVICE_FUNC inline int convert(int in, int dim) const
      {
        int res = in;
        static_assert ( A == Area::Local || A == Area::Global);
        if constexpr(A == Area::Local)
        {
          /** Shift the point **/
          res = res - offset[dim];
        }

        if constexpr(A == Area::Global)
        {
          /** Shift the point **/
          res = res + offset[dim];
        }
        return res;
      }

    template<Area A>
      ONIKA_HOST_DEVICE_FUNC onika::math::Vec3d compute_position(int x, int y, int z) const
      {
        static_assert(A == Area::Global);
        onika::math::Vec3d res = {(double)(x + offset[0]),(double)(y + offset[1]),(double)(z + offset[2])};
        res = {res.x * dx, res.y * dx, res.z * dx}; // add operator *=
        return res;
      }

    /**
     * @brief Computes the physical coordinates of a grid point from a linear index.
     *
     * Converts a linear grid index `id` into physical coordinates
     * in the global coordinate system.
     *
     * @tparam A Must be `Area::Global` (static_assert enforced).
     *
     * @param id Linear index of the grid point.
     * @return Vec3d The physical coordinates of the point.
     */
    template<Area A>
      ONIKA_HOST_DEVICE_FUNC onika::math::Vec3d compute_position(int id) const
      {
        static_assert(A == Area::Global);
        auto [x,y,z] = this->operator()(id);
        onika::math::Vec3d res = {(double)(x + offset[0]),(double)(y + offset[1]),(double)(z + offset[2])};
        res = {res.x * dx, res.y * dx, res.z * dx}; // add operator *=
        return res;
      }

    /**
     * @brief Restricts an input box to the current grid subdomain.
     *
     * Converts the input box to local or global coordinates (depending on `A`),
     * then intersects it with the subdomain defined by `Traversal Tr`.
     *
     * @tparam A Area type: Local or Global.
     * @tparam Tr Traversal type: defines which part of the grid to consider.
     *
     * @param input_box The input Box3D to restrict.
     * @return std::tuple<bool, Box3D> 
     *         - bool: true if there is any intersection with the subdomain.
     *         - Box3D: the adjusted box clipped to the grid subdomain.
     */
    template<Area A, Traversal Tr>
      ONIKA_HOST_DEVICE_FUNC std::tuple<bool, Box3D> restrict_box_to_grid(const Box3D& input_box) const
      {
        // Convert input box to local/global coordinates
        Box3D adjusted_box;
        adjusted_box.inf = convert<A, false>(input_box.inf);
        adjusted_box.sup = convert<A, false>(input_box.sup);
        Box3D subdomain = build_box<A,Tr>();

        // Check if there is any intersection
        bool is_inside_subdomain = intersect(subdomain, adjusted_box);
        if( ! is_inside_subdomain ) 
        {
          return {false, adjusted_box};
        }

        // Clip the box to the subdomain boundaries
        for(int dim = 0; dim < 3 ; dim++)
        {
          adjusted_box.inf[dim] = std::max(adjusted_box.inf[dim], subdomain.inf[dim]);
          adjusted_box.sup[dim] = std::min(adjusted_box.sup[dim], subdomain.sup[dim]);
        }
        return {true, adjusted_box};
      }


    ONIKA_HOST_DEVICE_FUNC int operator()(Point3D& p) const {  return bx(p[0], p[1], p[2]); }  
    ONIKA_HOST_DEVICE_FUNC int operator()(Point3D&& p) const { return bx(p[0], p[1], p[2]); }  
    ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const { return bx(x, y, z); }  
    ONIKA_HOST_DEVICE_FUNC inline std::tuple<int,int,int> operator()(int idx) const {  return bx(idx);  }  
  };

  /**
   * @brief Helper struct to convert between 3D grid coordinates and linear indices.
   * 
   * This is used locally within a grid to map (i,j,k) coordinates to a linear index and vice versa.
   */
  struct GridIKJtoIdx
  {
    Box3D bx; ///< The underlying 3D box for indexing

    GridIKJtoIdx() = delete;
    GridIKJtoIdx(const LBMGrid& grid) { bx = grid.bx; }
    GridIKJtoIdx(const Box3D& in) { bx = in; }
    ONIKA_HOST_DEVICE_FUNC int operator()(Point3D& p) const  { return bx(p[0], p[1], p[2]); }  
    ONIKA_HOST_DEVICE_FUNC int operator()(Point3D&& p) const { return bx(p[0], p[1], p[2]); }  
    ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const { return bx(x, y, z); }  
    ONIKA_HOST_DEVICE_FUNC inline std::tuple<int,int,int> operator()(int idx) const { return bx(idx);  }  
  };
}
