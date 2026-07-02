#pragma once
#include "stencil.hpp"

namespace hippoLBM::stencil {

// do [Start, Start+Step, ..., End[
template <int Start, int End, int Step, int... Is>
struct FilteredSequence {
  using type = typename FilteredSequence<Start + Step, End, Step, Is..., Start>::type;
};
template <int End, int Step, int... Is>
struct FilteredSequence<End, End, Step, Is...> {
  using type = std::integer_sequence<int, Is...>;
};

template <int Start, int End, int Step>
using make_filtered_sequence = typename FilteredSequence<Start, End, Step>::type;

template <typename Stencil, typename F, int... Is>
ONIKA_HOST_DEVICE_FUNC void for_specific_dirs_impl(F&& f, std::integer_sequence<int, Is...>) {
  (f.template operator()<typename Stencil::template dir<Is>>(Is), ...);
}

template <typename Stencil, int Start, int End, int Step, typename F, int... Is>
ONIKA_HOST_DEVICE_FUNC void for_each_impl(F&& f, std::integer_sequence<int, Is...>) {
  (f.template operator()<typename Stencil::template dir<Is>>(Is), ...);
}

template <typename LBMScheme, int Start = 0, int End = LBMScheme::Q, int Step = 1, typename F>
ONIKA_HOST_DEVICE_FUNC void for_each(F&& f) {
  for_each_impl<LBMScheme, Start, End, Step>(std::forward<F>(f), make_filtered_sequence<Start, End, Step>{});
}

}  // namespace hippoLBM::stencil