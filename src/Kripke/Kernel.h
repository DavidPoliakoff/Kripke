//
// Copyright (c) 2014-19, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#ifndef KRIPKE_KERNEL_H__
#define KRIPKE_KERNEL_H__

#include <Kripke.h>
#include <Kripke/Core/DataStore.h>
#include <utility>
#include <type_traits>

template<typename... X>
struct is_lambda_execution : public std::false_type {
    static constexpr bool value = false;
};

template<int N>
struct is_lambda_execution<RAJA::statement::Lambda<N>> : public std::true_type {
    static constexpr bool value = true;
};

template<typename... X>
struct is_nested_for : public std::false_type {
    static constexpr bool value = false;
};

template<camp::idx_t N, class EP, class... Statements>
struct is_nested_for<RAJA::statement::For<N,EP,Statements...>> : public std::true_type {
    static constexpr bool value = is_nested_for<Statements...>::value || 
    is_lambda_execution<Statements...>::value;
};
template<typename... X>
struct dbg : public std::false_type {
    static constexpr bool value = false;
};

template<camp::idx_t N, typename EP, typename... Statements>
struct dbg<RAJA::statement::For<N,EP,Statements...>> : public std::true_type {
    static constexpr bool value = true;
};

struct nonesuch {
  nonesuch()                = delete;
  ~nonesuch()               = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

namespace detail {

template<class... T>
struct void_t { using type = void; };       


template<class Default, class Void, template<class...> class Op, class... Args>
struct ListChecker{
        using value_t = std::false_type;
        using type = Default;
};

template <class Default, template<class...> class Op, class... Args>
struct ListChecker<Default, void_t<Op<camp::list<Args...>>>, Op, Args...> {
  using value_t = std::true_type;
  using type = Op<Args...>;
};

} // end namespace detail

template <template<class...> class Op, class... Args>
using list_is = typename detail::ListChecker<nonesuch, void, Op, Args...>::value_t;

template<class Y,class... X>
void kernel_wrapper(X... x);

//template<class Y,class... X>
//auto kernel_wrapper(X... x) -> typename std::enable_if<list_is<is_nested_for,Y>::value_t,void>::type  {
//#ifndef KRIPKE_USE_KOKKOS
//  RAJA::kernel<Y>(std::forward<X>(x)...);        
//#endif
//}

namespace impl {
  template<class Policy, size_t... Indices>
  struct list_adder;

  template<std::size_t Idx, size_t... Indices>
  struct list_adder<RAJA::statement::Lambda<Idx>, Indices...>{
    using ordering = std::index_sequence<Indices...>;
  };

  template<std::size_t Idx, class Policy, class... Statements, size_t... Indices>
  struct list_adder<RAJA::statement::For<Idx, Policy, Statements...>, Indices...> {
    using ordering = typename list_adder<Statements..., Indices..., Idx>::ordering;
  };
   
  template<std::size_t... N, class... Segments, class Callable>
  void kokkos_launcher(std::index_sequence<N...>, camp::tuple<Segments...>&& segments, Callable callable) {
  //Kokkos::MDRangePolicy<Kokkos::Rank<sizeof...(N) , Kokkos::Iterate::Left, Kokkos::Iterate::Right >> policy(
  Kokkos::MDRangePolicy<Kokkos::Rank<sizeof...(N) >> policy(
                  //{std::begin(camp::get<N>(segments))...},
                  //{std::end(camp::get<N>(segments))...} 
                  {**std::begin(camp::get<N>(segments))...},
                  {**std::end(camp::get<N>(segments))...} 
                  );
 // Kokkos::parallel_for(policy, args...);
  
  Kokkos::parallel_for(policy, [&](
   typename std::remove_reference<decltype(**std::declval<Segments>().begin())>::type... segs
                          ) {
    //callable(decltype(*std::declval<Segments>().begin())(segs)...);
    callable(decltype(*std::declval<Segments>().begin())(segs)...);
  });
  Kokkos::fence();
}
}
template<std::size_t... ordering, class... Segments, class... Args>
void kokkos_launcher(std::index_sequence<ordering...> order_sequence, camp::tuple<Segments...>&& segments, Args... args){
  impl::kokkos_launcher(order_sequence, std::forward<camp::tuple<Segments...>>(segments), std::forward<Args...>(args...));
}


template<class Y, class Tuple, class... X>
void kernel_wrapper(Tuple t, X... x){
  //static_assert(list_is<is_nested_for,Y>::value, "args to kernel wrapper must be nested for");
#ifndef KRIPKE_USE_KOKKOS
  RAJA::kernel<Y>(std::forward<Tuple>(t), std::forward<X>(x)...);        
#else
  //RAJA::kernel<Y>(std::forward<Tuple>(t), std::forward<X>(x)...);        
  using inner_policy = camp::at<Y,camp::num<0>>;
  kokkos_launcher(typename impl::list_adder<typename inner_policy::type>::ordering{},std::forward<Tuple>(t), std::forward<X>(x)...); 
#endif
}

template<class... Y,class Segment,class Callable>
void forall_wrapper(Segment&& seg, Callable&& callable){
#ifndef KRIPKE_USE_KOKKOS
  RAJA::forall<Y...>(seg, callable);        
#else
  RAJA::forall<Y...>(std::forward<Segment>(seg), std::forward<Callable>(callable));        
  //const long begin = **seg.begin();
  //const long end = **seg.end();
  //Kokkos::RangePolicy<> policy(begin,end);
  //Kokkos::parallel_for(policy, [&](const unsigned long in){
  //});
#endif
}

namespace Kripke {

  namespace Kernel {

    void LPlusTimes(Kripke::Core::DataStore &data_store);


    void LTimes(Kripke::Core::DataStore &data_store);


    double population(Kripke::Core::DataStore &data_store);


    void scattering(Kripke::Core::DataStore &data_store);


    void source(Kripke::Core::DataStore &data_store);


    void sweepSubdomain(Kripke::Core::DataStore &data_store, Kripke::SdomId sdom_id);


    template<typename FieldType>
    RAJA_INLINE
    void kConst(FieldType &field, Kripke::SdomId sdom_id, typename FieldType::ElementType value){
      auto view1d = field.getView1d(sdom_id);
      int num_elem = field.size(sdom_id);
      RAJA::forall<RAJA::loop_exec>(
        RAJA::RangeSegment(0, num_elem),
        [=](RAJA::Index_type i){
			 	  view1d(i) = value;
      });
    }

    template<typename FieldType>
    RAJA_INLINE
    void kConst(FieldType &field, typename FieldType::ElementType value){
      for(Kripke::SdomId sdom_id : field.getWorkList()){
        kConst(field, sdom_id, value);
      }
    }




    template<typename FieldType>
    RAJA_INLINE
    void kCopy(FieldType &field_dst, Kripke::SdomId sdom_id_dst,
               FieldType &field_src, Kripke::SdomId sdom_id_src){
      auto view_src = field_src.getView1d(sdom_id_src);
      auto view_dst = field_dst.getView1d(sdom_id_dst);
      int num_elem = field_src.size(sdom_id_src);

      RAJA::forall<RAJA::loop_exec>(
        RAJA::RangeSegment(0, num_elem),
        [=](RAJA::Index_type i){
          view_src(i) = view_dst(i);
      });
    }

    template<typename FieldType>
    RAJA_INLINE
    void kCopy(FieldType &field_dst, FieldType &field_src){
      for(Kripke::SdomId sdom_id : field_dst.getWorkList()){
        kCopy(field_dst, sdom_id, field_src, sdom_id);
      }
    }

  }
}

#endif
