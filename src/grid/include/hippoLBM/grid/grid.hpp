#pragma once

#include <onika/math/basic_types.h>

#include <hippoLBM/core/box3d.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/core/point3d.hpp>

namespace hippoLBM {
/**
 * @brief Lattice Boltzmann Method grid representation.
 * Stores the local grid, extended grid, offset_, ghost layers, and spatial resolution.
 */
struct LBMGrid {
  static constexpr int DIM = 3;  ///< Number of spatial dimensions

  Box3D bx_;                   ///< Box covering the actual computational grid
  Box3D ext_;                  ///< Box covering the extended grid (includes ghost layers)
  Point3D offset_;             ///< Offset of the real grid (useful for global indexing)
  int ghost_layer_ = 2;        ///< Number of ghost layers (default 2 for DEM + LBM)
  double dx_;                  ///< Distance between two grid points (spatial resolution)
  onika::math::Vec3d origin_;  ///< Origin of the domain (not the subdomain)

  LBMGrid() {};

  LBMGrid(Box3D& b, Point3D& o, const int g, double d, onika::math::Vec3d& origin)
      : bx_(b), offset_(o), ghost_layer_(g), dx_(d), origin_(origin) {
    // add a print function here ?
  }

  // setter
  inline void set_box(Box3D& b) { bx_ = b; }
  inline void set_ext(Box3D& e) { ext_ = e; }
  inline void set_offset(Point3D& o) { offset_ = o; }
  inline void set_offset(Point3D&& o) { offset_ = o; }
  inline void set_ghost_layer(const int g) { ghost_layer_ = g; }
  inline void set_dx(const double d) { dx_ = d; }
  inline void set_origin(const onika::math::Vec3d& origin) { origin_ = origin; }

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
   * - If `Area::Global`, the global offset_ is added.
   */
  template <Area A, Traversal Tr>
  ONIKA_HOST_DEVICE_FUNC inline int start(const int dim) const {
    // static_assert(A == Area::Local);
    int res = 0;
    static_assert(Tr == Traversal::All || Tr == Traversal::Real || Tr == Traversal::Inside || Tr == Traversal::Extend);
    if constexpr (Tr == Traversal::All) res = bx_.start(dim);
    if constexpr (Tr == Traversal::Real) res = bx_.start(dim) + ghost_layer_;
    if constexpr (Tr == Traversal::Inside) res = bx_.start(dim) + ghost_layer_ + 1;
    if constexpr (Tr == Traversal::Extend) res = ext_.start(dim);
    if constexpr (A == Area::Global) res += offset_[dim];
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
   * - If `Area::Global`, the global offset_ is added.
   */
  template <Area A, Traversal Tr>
  ONIKA_HOST_DEVICE_FUNC inline int end(const int dim) const {
    // static_assert(A == Area::Local);
    static_assert(Tr == Traversal::All || Tr == Traversal::Real || Tr == Traversal::Inside || Tr == Traversal::Extend);

    int res = 0;
    if constexpr (Tr == Traversal::All) res = bx_.end(dim);
    if constexpr (Tr == Traversal::Real) res = bx_.end(dim) - ghost_layer_;
    if constexpr (Tr == Traversal::Inside) res = bx_.end(dim) - ghost_layer_ - 1;
    if constexpr (Tr == Traversal::Extend) res = ext_.end(dim);
    if constexpr (A == Area::Global) res += offset_[dim];
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
   * - `(Area::Local, Traversal::All)` returns the full local box (`bx_`).
   * - `(Area::Local, Traversal::Extend)` returns the extended box (`ext_`).
   * - For other cases, the box is built from `start<A,Tr>(dim)` and `end<A,Tr>(dim)` for each dimension.
   */
  template <Area A, Traversal Tr>
  ONIKA_HOST_DEVICE_FUNC Box3D build_box() const {
    if constexpr (A == Area::Local && Tr == Traversal::All) return bx_;
    if constexpr (A == Area::Local && Tr == Traversal::Extend) return ext_;
    Point3D lower, upper;
    for (int dim = 0; dim < DIM; dim++) {
      lower[dim] = this->start<A, Tr>(dim);
      upper[dim] = this->end<A, Tr>(dim);
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
  template <Area A, Traversal Tr>
  ONIKA_HOST_DEVICE_FUNC inline bool contains(Point3D& p) const {
    for (int dim = 0; dim < DIM; dim++) {
      if (p[dim] < this->start<A, Tr>(dim) || p[dim] > this->end<A, Tr>(dim)) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Checks whether a point lies inside the extended grid region.
   *
   * A point is considered defined if all its coordinates are within the
   * lower and upper bounds of the extended box (`ext_`).
   *
   * @param p Point coordinates to be tested.
   * @return true if the point is inside the extended region, false otherwise.
   *
   * @note
   * - The lower bound (`ext_.start(dim)`) is inclusive.
   * - The upper bound (`ext_.end(dim)`) is also inclusive.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool is_defined(Point3D& p) const {
    for (int dim = 0; dim < DIM; dim++) {
      if (p[dim] < ext_.start(dim) || p[dim] > ext_.end(dim)) {
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
   * - The lower bound (`ext_.start(dim)`) is inclusive.
   * - The upper bound (`ext_.end(dim)`) is also inclusive.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool is_defined(int i, int j, int k) const {
    if (i < ext_.start(0) || i > ext_.end(0)) return false;
    if (j < ext_.start(1) || j > ext_.end(1)) return false;
    if (k < ext_.start(2) || k > ext_.end(2)) return false;
    return true;
  }

  /**
   * @brief Checks whether a point lies inside the local grid region.
   *
   * A point is considered local if all its coordinates fall within the
   * bounds of the local box (`bx_`).
   *
   * @param p Point coordinates to be tested.
   * @return true if the point is inside the local region, false otherwise.
   *
   * @note
   * - The lower bound (`bx_.start(dim)`) is inclusive.
   * - The upper bound (`bx_.end(dim)`) is also inclusive.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool is_local(Point3D& p) const {
    for (int dim = 0; dim < DIM; dim++) {
      if (p[dim] < bx_.start(dim) || p[dim] > bx_.end(dim)) return false;
    }
    return true;
  }

  /**
   * @brief Checks whether a point lies inside the global grid region.
   *
   * @param p Point in global coordinates.
   * @return true if the point belongs to the global region, false otherwise.
   * @note
   * - Internally, this computes `local = p + offset_` and calls `is_local(local)`.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool is_global(Point3D& p) {
    Point3D local = p + offset_;
    return is_local(local);
  }

  /**
   * @brief Converts a point between local and global coordinate systems
   *
   * @tparam A     Conversion mode: `Area::Local` or `Area::Global`.
   * @tparam Check If true, performs an assertion to ensure the point lies
   *               inside the valid grid region before conversion.
   * @param x Coordinate along the X axis.
   * @param y Coordinate along the Y axis.
   * @param z Coordinate along the Z axis.
   * @return A Point3D in the target coordinate system.
   */
  template <Area A, bool Check = false>
  ONIKA_HOST_DEVICE_FUNC inline Point3D convert(int x, int y, int z) const {
    static_assert(A == Area::Local || A == Area::Global || A == Area::AsIs);

    Point3D res = {x, y, z};
    if constexpr (A == Area::AsIs) {
      return res;
    }
    if constexpr (A == Area::Local) {
      /** Convert global → local **/
      res = res - offset_;
      /** Optional runtime check: point must be inside local domain **/
      if constexpr (Check) assert(this->is_local(res));
    }

    if constexpr (A == Area::Global) {
      /** Optional runtime check: point must be valid before conversion **/
      if constexpr (Check) assert(this->is_local(res));
      /** Convert local → global **/
      res = res + offset_;
    }
    return res;
  }

  /*** convert a point to A area. */
  template <Area A, bool Check = false>
  ONIKA_HOST_DEVICE_FUNC inline Point3D convert(Point3D p) const {
    return convert<A, Check>(p[0], p[1], p[2]);
  }

  template <Area A, bool Check = false>
  ONIKA_HOST_DEVICE_FUNC inline Point3D convert(std::tuple<int, int, int>&& p) const {
    return convert<A, Check>(std::get<0>(p), std::get<1>(p), std::get<2>(p));
  }

  /*** @brief convert a point to A area. */
  template <Area A>
  ONIKA_HOST_DEVICE_FUNC inline int convert(int in, int dim) const {
    static_assert(A != Area::Undefined);
    if constexpr (A == Area::AsIs) {
      return in;
    }

    int res = in;

    if constexpr (A == Area::Local) {
      /** Shift the point **/
      res = res - offset_[dim];
    }

    if constexpr (A == Area::Global) {
      /** Shift the point **/
      res = res + offset_[dim];
    }
    return res;
  }

  /*** convert a box to A area. */
  template <Area A, bool Check = false>
  ONIKA_HOST_DEVICE_FUNC inline Box3D convert(Box3D box) const {
    Box3D res;
    res.inf_ = convert<A, Check>(box.inf_);
    res.sup_ = convert<A, Check>(box.sup_);
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
   * - Each coordinate is divided by `dx_` and truncated to an integer.
   * - The result is then passed to `convert<A,Check>` to handle the
   *   local/global conversion and optional validity check.
   */
  template <Area A, bool Check = false>
  ONIKA_HOST_DEVICE_FUNC inline Point3D project_to_grid(const onika::math::Vec3d&& r) const {
    onika::math::Vec3d proj = (r - origin_) / dx_;
    Point3D p = Point3D{int(std::floor(proj.x)), int(std::floor(proj.y)), int(std::floor(proj.z))};
    if constexpr (A == Area::Global) {  // Already in the global area
      return p;
    } else {
      return convert<Area::Local, Check>(p[0], p[1], p[2]);
    }
  }

  template <Area A, bool Check = false>
  ONIKA_HOST_DEVICE_FUNC inline Point3D project_to_grid(const onika::math::Vec3d& r) const {
    onika::math::Vec3d proj = (r - origin_) / dx_;
    Point3D p = Point3D{int(std::floor(proj.x)), int(std::floor(proj.y)), int(std::floor(proj.z))};
    if constexpr (A == Area::Global) {
      return p;
    } else {
      return convert<Area::Local, Check>(p[0], p[1], p[2]);
    }
  }

  template <Area A>
  ONIKA_HOST_DEVICE_FUNC onika::math::Vec3d compute_position(int x, int y, int z) const {
    static_assert(A == Area::Global || A == Area::AsIs);
    onika::math::Vec3d res = {double(x), double(y), double(z)};
    if constexpr (A == Area::Global) res += offset_;
    res = {res.x * dx_, res.y * dx_, res.z * dx_};  // add operator *=
    return res;
  }

  template <Area A>
  ONIKA_HOST_DEVICE_FUNC onika::math::Vec3d compute_position(Point3D&& pt) const {
    static_assert(A == Area::Global || A == Area::AsIs);

    return compute_position<A>(pt[0], pt[1], pt[2]);
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
  template <Area A>
  ONIKA_HOST_DEVICE_FUNC onika::math::Vec3d compute_position(int id) const {
    static_assert(A == Area::Global || A == Area::AsIs);

    Point3D pt = this->operator()(id);
    onika::math::Vec3d res = pt;
    if constexpr (A == Area::Global) res += offset_;
    res *= dx_;
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
  template <Area A, Traversal Tr, Area InputBoxArea = A>
  ONIKA_HOST_DEVICE_FUNC std::tuple<bool, Box3D> restrict_box_to_grid(const Box3D& input_box) const {
    // Convert input box to local/global coordinates
    Box3D adjusted_box;

    adjusted_box.inf_ = convert<InputBoxArea, false>(input_box.inf_);
    adjusted_box.sup_ = convert<InputBoxArea, false>(input_box.sup_);

    Box3D subdomain = build_box<A, Tr>();

    // Check if there is any intersection
    bool is_inside_subdomain = intersect(subdomain, adjusted_box);
    if (!is_inside_subdomain) {
      return {false, adjusted_box};
    }

    // Clip the box to the subdomain boundaries
    for (int dim = 0; dim < 3; dim++) {
      adjusted_box.inf_[dim] = std::max(adjusted_box.inf_[dim], subdomain.inf_[dim]);
      adjusted_box.sup_[dim] = std::min(adjusted_box.sup_[dim], subdomain.sup_[dim]);
    }
    return {true, adjusted_box};
  }

  ONIKA_HOST_DEVICE_FUNC int operator()(Point3D& p) const { return bx_(p[0], p[1], p[2]); }
  ONIKA_HOST_DEVICE_FUNC int operator()(Point3D&& p) const { return bx_(p[0], p[1], p[2]); }
  ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const { return bx_(x, y, z); }
  ONIKA_HOST_DEVICE_FUNC inline Point3D operator()(int idx) const { return bx_(idx); }
};

/**
 * @brief Helper struct to convert between 3D grid coordinates and linear indices.
 *
 * This is used locally within a grid to map (i,j,k) coordinates to a linear index and vice versa.
 */
struct GridIKJtoIdx {
  Box3D bx_;  ///< The underlying 3D box for indexing

  GridIKJtoIdx() = delete;
  GridIKJtoIdx(const LBMGrid& grid) { bx_ = grid.bx_; }
  GridIKJtoIdx(const Box3D& in) { bx_ = in; }
  ONIKA_HOST_DEVICE_FUNC int operator()(Point3D& p) const { return bx_(p[0], p[1], p[2]); }
  ONIKA_HOST_DEVICE_FUNC int operator()(Point3D&& p) const { return bx_(p[0], p[1], p[2]); }
  ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const { return bx_(x, y, z); }
  ONIKA_HOST_DEVICE_FUNC inline Point3D operator()(int idx) const { return bx_(idx); }
};
}  // namespace hippoLBM
