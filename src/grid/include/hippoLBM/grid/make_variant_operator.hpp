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

#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <yaml-cpp/yaml.h>
#include <memory>

#include <onika/type_utils.h>
#include <onika/log.h>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  /*
     Internal template utilities
   */
  namespace details
  {
    template< template<int> typename _OperatorTemplate, int... Q >
      struct MakeVariantOperatorHelper
      {
        static inline std::shared_ptr<onika::scg::OperatorNode> make_operator( const YAML::Node& node, const onika::scg::OperatorNodeFlavor& flavor )
        {
          return make_compatible_operator < _OperatorTemplate<Q>...> (node,flavor);
        }
      };

    template< template<int> class _OperatorTemplate, int... Q>
      struct make_variant_operator_t
      {
        static inline onika::scg::OperatorNodeCreateFunction make_factory(const std::string& opname)
        {
          onika::scg::OperatorNodeCreateFunction factory = [] (const YAML::Node& node, const onika::scg::OperatorNodeFlavor& flavor) -> std::shared_ptr<onika::scg::OperatorNode>
          {
            std::shared_ptr<onika::scg::OperatorNode> op = MakeVariantOperatorHelper< _OperatorTemplate, Q... >::make_operator(node,flavor);
            return op;        
          };

          return factory;
        }
      };

  } // temporary close exanb namespace
}

namespace onika
{
  namespace scg
  {
    template< template<int> class _OperatorTemplate, int... Q>
      struct OperatorNodeFactoryGenerator< hippoLBM::details::make_variant_operator_t<_OperatorTemplate, Q...> >
      {
        static inline OperatorNodeCreateFunction make_factory(const std::string& opname)
        {
          return hippoLBM::details::make_variant_operator_t<_OperatorTemplate, Q...>::make_factory(opname) ;
        }
      };
  }
}

namespace hippoLBM
{
  template< template<int> class _OperatorTemplate > 
    static inline constexpr 
    onika::scg::OperatorNodeFactoryGenerator< details::make_variant_operator_t< _OperatorTemplate, /*15, 27,*/ 19 > > make_variant_operator = {};
}

