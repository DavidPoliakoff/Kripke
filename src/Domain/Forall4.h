//AUTOGENERATED BY genForallN.py
  
#ifndef RAJA_DOMAIN_FORALL4_HXX__
#define RAJA_DOMAIN_FORALL4_HXX__

#include<RAJA/RAJA.hxx>
#include<Domain/Tile.h>

namespace RAJA {



/******************************************************************
 *  Policy base class, forall4()
 ******************************************************************/

// Execute (Termination default)
struct Forall4_Execute_Tag {};
struct Forall4_Execute {
  typedef Forall4_Execute_Tag PolicyTag;
};

// Starting (outer) policy for all forall4 policies
template<typename POL_I=RAJA::seq_exec, typename POL_J=RAJA::seq_exec, typename POL_K=RAJA::seq_exec, typename POL_L=RAJA::seq_exec, typename NEXT=Forall4_Execute>
struct Forall4_Policy {
  typedef NEXT NextPolicy;
  typedef POL_I PolicyI;
  typedef POL_J PolicyJ;
  typedef POL_K PolicyK;
  typedef POL_L PolicyL;
};

// Interchange loop order given permutation
struct Forall4_Permute_Tag {};
template<typename LOOP_ORDER, typename NEXT=Forall4_Execute>
struct Forall4_Permute {
  typedef Forall4_Permute_Tag PolicyTag;
  typedef NEXT NextPolicy;
  typedef LOOP_ORDER LoopOrder;
};

// Begin OpenMP Parallel Region
struct Forall4_OMP_Parallel_Tag {};
template<typename NEXT=Forall4_Execute>
struct Forall4_OMP_Parallel {
  typedef Forall4_OMP_Parallel_Tag PolicyTag;
  typedef NEXT NextPolicy;
};

// Tiling Policy
struct Forall4_Tile_Tag {};
template<typename TILE_I, typename TILE_J, typename TILE_K, typename TILE_L, typename NEXT=Forall4_Execute>
struct Forall4_Tile {
  typedef Forall4_Tile_Tag PolicyTag;
  typedef NEXT NextPolicy;
  typedef TILE_I TileI;
  typedef TILE_J TileJ;
  typedef TILE_K TileK;
  typedef TILE_L TileL;
};


/******************************************************************
 *  forall4_policy() Foreward declarations
 ******************************************************************/

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_Execute_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body);

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_Permute_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body);

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_OMP_Parallel_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body);

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_Tile_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body);


/******************************************************************
 *  Forall4Executor(): Default Executor for loops
 ******************************************************************/

template<typename POLICY_I, typename POLICY_J, typename POLICY_K, typename POLICY_L, typename TI, typename TJ, typename TK, typename TL>
struct Forall4Executor {
  template<typename BODY>
  inline void operator()(TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body) const {
    RAJA::forall<POLICY_I>(is_i, RAJA_LAMBDA(int i){
      exec(is_j, is_k, is_l, RAJA_LAMBDA(int j, int k, int l){
        body(i, j, k, l);
      });
    });
  }

  private:
    Forall3Executor<POLICY_J, POLICY_K, POLICY_L, TJ, TK, TL> exec;
};


/******************************************************************
 *  OpenMP Auto-Collapsing Executors for forall4()
 ******************************************************************/

#ifdef _OPENMP

// OpenMP Executor with collapse(2) for omp_parallel_for_exec
template<typename POLICY_K, typename POLICY_L, typename TK, typename TL>
class Forall4Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_K, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, TK, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, TL const &is_l, BODY body) const {
      int const i_start = is_i.getBegin();
      int const i_end   = is_i.getEnd();

      int const j_start = is_j.getBegin();
      int const j_end   = is_j.getEnd();

#pragma omp parallel for schedule(static) collapse(2)
      for(int i = i_start;i < i_end;++ i){
        for(int j = j_start;j < j_end;++ j){
          exec(is_k, is_l, RAJA_LAMBDA(int k, int l){
            body(i, j, k, l);
          });
      } } 
    }

  private:
    Forall2Executor<POLICY_K, POLICY_L, TK, TL> exec;
};

// OpenMP Executor with collapse(3) for omp_parallel_for_exec
template<typename POLICY_L, typename TL>
class Forall4Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, TL const &is_l, BODY body) const {
      int const i_start = is_i.getBegin();
      int const i_end   = is_i.getEnd();

      int const j_start = is_j.getBegin();
      int const j_end   = is_j.getEnd();

      int const k_start = is_k.getBegin();
      int const k_end   = is_k.getEnd();

#pragma omp parallel for schedule(static) collapse(3)
      for(int i = i_start;i < i_end;++ i){
        for(int j = j_start;j < j_end;++ j){
          for(int k = k_start;k < k_end;++ k){
            RAJA::forall<POLICY_L>(is_l, RAJA_LAMBDA(int l){
              body(i, j, k, l);
            });
      } } } 
    }
};

// OpenMP Executor with collapse(4) for omp_parallel_for_exec
template<>
class Forall4Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, BODY body) const {
      int const i_start = is_i.getBegin();
      int const i_end   = is_i.getEnd();

      int const j_start = is_j.getBegin();
      int const j_end   = is_j.getEnd();

      int const k_start = is_k.getBegin();
      int const k_end   = is_k.getEnd();

      int const l_start = is_l.getBegin();
      int const l_end   = is_l.getEnd();

#pragma omp parallel for schedule(static) collapse(4)
      for(int i = i_start;i < i_end;++ i){
        for(int j = j_start;j < j_end;++ j){
          for(int k = k_start;k < k_end;++ k){
            for(int l = l_start;l < l_end;++ l){
              body(i, j, k, l);
      } } } } 
    }
};

// OpenMP Executor with collapse(2) for omp_for_nowait_exec
template<typename POLICY_K, typename POLICY_L, typename TK, typename TL>
class Forall4Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_K, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, TK, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, TL const &is_l, BODY body) const {
      int const i_start = is_i.getBegin();
      int const i_end   = is_i.getEnd();

      int const j_start = is_j.getBegin();
      int const j_end   = is_j.getEnd();

#pragma omp for schedule(static) collapse(2) nowait
      for(int i = i_start;i < i_end;++ i){
        for(int j = j_start;j < j_end;++ j){
          exec(is_k, is_l, RAJA_LAMBDA(int k, int l){
            body(i, j, k, l);
          });
      } } 
    }

  private:
    Forall2Executor<POLICY_K, POLICY_L, TK, TL> exec;
};

// OpenMP Executor with collapse(3) for omp_for_nowait_exec
template<typename POLICY_L, typename TL>
class Forall4Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, TL const &is_l, BODY body) const {
      int const i_start = is_i.getBegin();
      int const i_end   = is_i.getEnd();

      int const j_start = is_j.getBegin();
      int const j_end   = is_j.getEnd();

      int const k_start = is_k.getBegin();
      int const k_end   = is_k.getEnd();

#pragma omp for schedule(static) collapse(3) nowait
      for(int i = i_start;i < i_end;++ i){
        for(int j = j_start;j < j_end;++ j){
          for(int k = k_start;k < k_end;++ k){
            RAJA::forall<POLICY_L>(is_l, RAJA_LAMBDA(int l){
              body(i, j, k, l);
            });
      } } } 
    }
};

// OpenMP Executor with collapse(4) for omp_for_nowait_exec
template<>
class Forall4Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, BODY body) const {
      int const i_start = is_i.getBegin();
      int const i_end   = is_i.getEnd();

      int const j_start = is_j.getBegin();
      int const j_end   = is_j.getEnd();

      int const k_start = is_k.getBegin();
      int const k_end   = is_k.getEnd();

      int const l_start = is_l.getBegin();
      int const l_end   = is_l.getEnd();

#pragma omp for schedule(static) collapse(4) nowait
      for(int i = i_start;i < i_end;++ i){
        for(int j = j_start;j < j_end;++ j){
          for(int k = k_start;k < k_end;++ k){
            for(int l = l_start;l < l_end;++ l){
              body(i, j, k, l);
      } } } } 
    }
};


#endif // _OPENMP


/******************************************************************
 *  forall4_permute(): Permutation function overloads
 ******************************************************************/

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_IJKL, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyI, PolicyJ, PolicyK, PolicyL>(NextPolicyTag(), is_i, is_j, is_k, is_l,
    RAJA_LAMBDA(int i, int j, int k, int l){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_IJLK, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyI, PolicyJ, PolicyL, PolicyK>(NextPolicyTag(), is_i, is_j, is_l, is_k,
    RAJA_LAMBDA(int i, int j, int l, int k){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_IKJL, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyI, PolicyK, PolicyJ, PolicyL>(NextPolicyTag(), is_i, is_k, is_j, is_l,
    RAJA_LAMBDA(int i, int k, int j, int l){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_IKLJ, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyI, PolicyK, PolicyL, PolicyJ>(NextPolicyTag(), is_i, is_k, is_l, is_j,
    RAJA_LAMBDA(int i, int k, int l, int j){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_ILJK, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyI, PolicyL, PolicyJ, PolicyK>(NextPolicyTag(), is_i, is_l, is_j, is_k,
    RAJA_LAMBDA(int i, int l, int j, int k){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_ILKJ, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyI, PolicyL, PolicyK, PolicyJ>(NextPolicyTag(), is_i, is_l, is_k, is_j,
    RAJA_LAMBDA(int i, int l, int k, int j){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_JIKL, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyJ, PolicyI, PolicyK, PolicyL>(NextPolicyTag(), is_j, is_i, is_k, is_l,
    RAJA_LAMBDA(int j, int i, int k, int l){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_JILK, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyJ, PolicyI, PolicyL, PolicyK>(NextPolicyTag(), is_j, is_i, is_l, is_k,
    RAJA_LAMBDA(int j, int i, int l, int k){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_JKIL, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyJ, PolicyK, PolicyI, PolicyL>(NextPolicyTag(), is_j, is_k, is_i, is_l,
    RAJA_LAMBDA(int j, int k, int i, int l){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_JKLI, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyJ, PolicyK, PolicyL, PolicyI>(NextPolicyTag(), is_j, is_k, is_l, is_i,
    RAJA_LAMBDA(int j, int k, int l, int i){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_JLIK, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyJ, PolicyL, PolicyI, PolicyK>(NextPolicyTag(), is_j, is_l, is_i, is_k,
    RAJA_LAMBDA(int j, int l, int i, int k){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_JLKI, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyJ, PolicyL, PolicyK, PolicyI>(NextPolicyTag(), is_j, is_l, is_k, is_i,
    RAJA_LAMBDA(int j, int l, int k, int i){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_KIJL, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyK, PolicyI, PolicyJ, PolicyL>(NextPolicyTag(), is_k, is_i, is_j, is_l,
    RAJA_LAMBDA(int k, int i, int j, int l){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_KILJ, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyK, PolicyI, PolicyL, PolicyJ>(NextPolicyTag(), is_k, is_i, is_l, is_j,
    RAJA_LAMBDA(int k, int i, int l, int j){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_KJIL, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyK, PolicyJ, PolicyI, PolicyL>(NextPolicyTag(), is_k, is_j, is_i, is_l,
    RAJA_LAMBDA(int k, int j, int i, int l){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_KJLI, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyK, PolicyJ, PolicyL, PolicyI>(NextPolicyTag(), is_k, is_j, is_l, is_i,
    RAJA_LAMBDA(int k, int j, int l, int i){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_KLIJ, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyK, PolicyL, PolicyI, PolicyJ>(NextPolicyTag(), is_k, is_l, is_i, is_j,
    RAJA_LAMBDA(int k, int l, int i, int j){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_KLJI, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyK, PolicyL, PolicyJ, PolicyI>(NextPolicyTag(), is_k, is_l, is_j, is_i,
    RAJA_LAMBDA(int k, int l, int j, int i){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_LIJK, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyL, PolicyI, PolicyJ, PolicyK>(NextPolicyTag(), is_l, is_i, is_j, is_k,
    RAJA_LAMBDA(int l, int i, int j, int k){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_LIKJ, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyL, PolicyI, PolicyK, PolicyJ>(NextPolicyTag(), is_l, is_i, is_k, is_j,
    RAJA_LAMBDA(int l, int i, int k, int j){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_LJIK, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyL, PolicyJ, PolicyI, PolicyK>(NextPolicyTag(), is_l, is_j, is_i, is_k,
    RAJA_LAMBDA(int l, int j, int i, int k){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_LJKI, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyL, PolicyJ, PolicyK, PolicyI>(NextPolicyTag(), is_l, is_j, is_k, is_i,
    RAJA_LAMBDA(int l, int j, int k, int i){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_LKIJ, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyL, PolicyK, PolicyI, PolicyJ>(NextPolicyTag(), is_l, is_k, is_i, is_j,
    RAJA_LAMBDA(int l, int k, int i, int j){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}

template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_permute(PERM_LKJI, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Call next policy with permuted indices and policies
  forall4_policy<NextPolicy, PolicyL, PolicyK, PolicyJ, PolicyI>(NextPolicyTag(), is_l, is_k, is_j, is_i,
    RAJA_LAMBDA(int l, int k, int j, int i){
      // Call body with non-permuted indices
      body(i, j, k, l);
    });
}


/******************************************************************
 *  forall4_policy() Policy Layer, overloads for policy tags
 ******************************************************************/


/**
 * Execute inner loops policy function.
 * This is the default termination case.
 */
    template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_Execute_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){

  // Create executor object to launch loops
  Forall4Executor<PolicyI, PolicyJ, PolicyK, PolicyL, TI, TJ, TK, TL> exec;

  // Launch loop body
  exec(is_i, is_j, is_k, is_l, body);
}


/**
 * Permutation policy function.
 * Provides loop interchange.
 */
    template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_Permute_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  // Get the loop permutation
  typedef typename POLICY::LoopOrder LoopOrder;

  // Call loop interchange overload to re-wrire indicies and policies
  forall4_permute<POLICY, PolicyI, PolicyJ, PolicyK, PolicyL>(LoopOrder(), is_i, is_j, is_k, is_l, body);
}


/**
 * OpenMP Parallel Region Section policy function.
 */
    template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_OMP_Parallel_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // create OpenMP Parallel Region
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // execute the next policy
    forall4_policy<NextPolicy, PolicyI, PolicyJ, PolicyK, PolicyL>(NextPolicyTag(), is_i, is_j, is_k, is_l, body);
  }
}


/**
 * Tiling policy function.
 */
    template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_Tile_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;
  typedef typename POLICY::TileI TileI;
  typedef typename POLICY::TileJ TileJ;
  typedef typename POLICY::TileK TileK;
  typedef typename POLICY::TileL TileL;

  // execute the next policy
      forall_tile(TileI(), is_i, [=](RAJA::RangeSegment is_ii){
        forall_tile(TileJ(), is_j, [=](RAJA::RangeSegment is_jj){
          forall_tile(TileK(), is_k, [=](RAJA::RangeSegment is_kk){
            forall_tile(TileL(), is_l, [=](RAJA::RangeSegment is_ll){
          forall4_policy<NextPolicy, PolicyI, PolicyJ, PolicyK, PolicyL>(NextPolicyTag(), is_ii, is_jj, is_kk, is_ll, body);
            });
          });
        });
      });
}



/******************************************************************
 * forall4(), User interface
 * Provides index typing, and initial nested policy unwrapping
 ******************************************************************/

template<typename POLICY, typename IdxI=int, typename IdxJ=int, typename IdxK=int, typename IdxL=int, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4(TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  typedef typename POLICY::PolicyI                PolicyI;
  typedef typename POLICY::PolicyJ                PolicyJ;
  typedef typename POLICY::PolicyK                PolicyK;
  typedef typename POLICY::PolicyL                PolicyL;

  // call 'policy' layer with next policy
  forall4_policy<NextPolicy, PolicyI, PolicyJ, PolicyK, PolicyL>(NextPolicyTag(), is_i, is_j, is_k, is_l, 
    [=](int i, int j, int k, int l){
      body(IdxI(i), IdxJ(j), IdxK(k), IdxL(l));
    }
  );
}



} // namespace RAJA
  
#endif

