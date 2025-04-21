#pragma once

#include <onika/math/basic_types_def.h>
#include <grid/enum.hpp>
#include <grid/point.hpp>
#include <grid/box.hpp>
#include <cassert>
#include <tuple>

namespace hippoLBM
{

  template<int DIM/*, bool perX, bool perY, bool perZ*/>
    struct grid
    {
      box<DIM> bx; /*** box covering the grid */
      box<DIM> ext; /*** box covering the extended real grid */
      point<DIM> offset; /*** offset of the real grid */
      int ghost_layer = 2; /*** default is 2 for dem + lbm */
      double dx; /*** distance between two points   */


      grid() {};

      grid(box<DIM>& b, point<DIM>& o, const int g, double d) : bx(b), offset(o), ghost_layer(g), dx(d)
      {
        // add a print function here ?
      }

      inline void set_box(box<DIM>& b) { bx = b;}
      inline void set_ext(box<DIM>& e) {ext = e;}
      inline void set_offset(point<DIM>& o) {offset = o;}
      inline void set_offset(point<DIM>&& o) {offset = o;}
      inline void set_ghost_layer(const int g) {ghost_layer = g;}
      inline void set_dx(const double d) {dx = d;}

      template<Area A, Traversal Tr>
        ONIKA_HOST_DEVICE_FUNC inline int start(const int dim)
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

      template<Area A, Traversal Tr>
        ONIKA_HOST_DEVICE_FUNC inline int end(const int dim)
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


      template<Area A, Traversal Tr>
        ONIKA_HOST_DEVICE_FUNC box<3> build_box()
        {
          if constexpr (A == Area::Local && Tr == Traversal::All) return bx;
          if constexpr (A == Area::Local && Tr == Traversal::Extend) return ext;
          point<3> lower, upper;
          for(int dim = 0; dim < DIM ; dim++)
          {
            lower[dim] = this->start<A,Tr>(dim);
            upper[dim] = this->end<A,Tr>(dim);
          }
          box<3> res = {lower, upper};
          return res;
        }


      /*** check is the point is in the grid */
      template<Area A, Traversal Tr>
        ONIKA_HOST_DEVICE_FUNC inline bool contains(point<DIM>& p) const
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

      /*** check if the point is defined */
      ONIKA_HOST_DEVICE_FUNC inline bool is_defined(point<DIM>& p) const
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

      ONIKA_HOST_DEVICE_FUNC inline bool is_defined(int i, int j, int k) const
      {
        if( i < ext.start(0) || i > ext.end(0) ) return false;
        if( j < ext.start(1) || j > ext.end(1) ) return false;
        if( k < ext.start(2) || k > ext.end(2) ) return false;
        return true;
      }

      /*** @brief Check if the point is in the local grid */ 
      ONIKA_HOST_DEVICE_FUNC inline bool is_local(point<DIM>& p)
      {
        for(int dim = 0 ; dim < DIM ; dim++)
        {
          if( p[dim] < bx.start(dim) || p[dim] > bx.end(dim) ) return false; 
        }
        return true;
      }

      ONIKA_HOST_DEVICE_FUNC inline bool is_global(point<DIM>& p)
      {
        point<DIM> local = p + offset;
        return is_local(local);
      }

      /*** @brief convert a point to A area. */
      template<Area A, bool Check=false>
        ONIKA_HOST_DEVICE_FUNC inline point<DIM> convert(int x, int y, int z)
        {
          point<DIM> res = {x, y, z};
          static_assert ( A == Area::Local || A == Area::Global);
          if constexpr(A == Area::Local)
          {
            /** Shift the point **/
            res = res - offset;
            /** We can convert points that are not in the grid **/
            if constexpr (Check)  assert(this->is_local(res));
          }

          if constexpr(A == Area::Global)
          {
            /** We can convert points that are not in the grid **/
            if constexpr (Check)  assert(this->is_local(res));
            /** Shift the point **/
            res = res + offset;
          }
          return res;
        }

      // could be optimized
      /*** convert a point to A area. */
      template<Area A, bool Check=false>
        ONIKA_HOST_DEVICE_FUNC inline point<DIM> convert(point<3> p)
        {
          return convert<A,Check>(p[0], p[1], p[2]);
        }

      /*** @brief convert a point to A area. */
      template<Area A>
        ONIKA_HOST_DEVICE_FUNC inline int convert(int in, int dim)
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
        ONIKA_HOST_DEVICE_FUNC onika::math::Vec3d compute_position(int x, int y, int z)
        {
          static_assert(A == Area::Global);
          onika::math::Vec3d res = {(double)(x + offset[0]),(double)(y + offset[1]),(double)(z + offset[2])};
          res = {res.x * dx, res.y * dx, res.z * dx}; // add operator *=
          return res;
        }

      template<Area A, Traversal Tr>
        ONIKA_HOST_DEVICE_FUNC std::tuple<bool, box<3>> restrict_box_to_grid(const box<3>& input_box)
        {
          box<3> adjusted_box;
          adjusted_box.inf = convert<A, false>(input_box.inf);
          adjusted_box.sup = convert<A, false>(input_box.sup);
          box<3> subdomain = build_box<A,Tr>();
          bool is_inside_subdomain = intersect(subdomain, adjusted_box);
          if( ! is_inside_subdomain ) 
          {
            return {false, adjusted_box};
          }

          for(int dim = 0; dim < 3 ; dim++)
          {
            adjusted_box.inf[dim] = std::max(adjusted_box.inf[dim], subdomain.inf[dim]);
            adjusted_box.sup[dim] = std::min(adjusted_box.sup[dim], subdomain.sup[dim]);
          }
          return {true, adjusted_box};
        }


      ONIKA_HOST_DEVICE_FUNC int operator()(point<3>& p) const
      {
        return bx(p[0], p[1], p[2]);
      }  

      ONIKA_HOST_DEVICE_FUNC int operator()(point<3>&& p) const
      {
        return bx(p[0], p[1], p[2]);
      }  

      ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const
      {
        return bx(x, y, z);
      }  

      ONIKA_HOST_DEVICE_FUNC inline std::tuple<int,int,int> operator()(int idx) const
      {
        return bx(idx);
      }  
    };
}
