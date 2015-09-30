//AUTOGENERATED BY genForallN.py
  
#ifndef __DOMAIN_FORALL2_H__
#define __DOMAIN_FORALL2_H__



    template<typename POLICY, typename BODY>
    inline void forall2(int end_i, int end_j, BODY const &body){
      typedef typename POLICY::layout L;
      forall2<POLICY, BODY>(L(), end_i, end_j, body);
    }

/******************************************************************
 *  Implementation for permutations of forall2()
 ******************************************************************/

      template<typename POLICY, typename BODY>
      inline void forall2(LAYOUT_IJ, int end_i, int end_j, BODY const &body){
        forall<typename POLICY::pol_i>(0, end_i, [=](int i){
          forall<typename POLICY::pol_j>(0, end_j, [=](int j){
            body(i, j);
          });
        });
      }

      template<typename POLICY, typename BODY>
      inline void forall2(LAYOUT_JI, int end_i, int end_j, BODY const &body){
        forall<typename POLICY::pol_j>(0, end_j, [=](int j){
          forall<typename POLICY::pol_i>(0, end_i, [=](int i){
            body(i, j);
          });
        });
      }


  
#endif

