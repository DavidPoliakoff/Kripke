#include<Kripke/Kernel.h>
#include<Kripke/Comm.h>
#include<Kripke/Grid.h>
#include<Kripke/User_Data.h>
#include<Kripke/LMat.h>
#include<Kripke/SubTVec.h>

#include<Kripke/Kernel/Kernel_3d_GDZ.h>
#include<Kripke/Kernel/Kernel_3d_DGZ.h>
#include<Kripke/Kernel/Kernel_3d_ZDG.h>
#include<Kripke/Kernel/Kernel_3d_DZG.h>
#include<Kripke/Kernel/Kernel_3d_ZGD.h>
#include<Kripke/Kernel/Kernel_3d_GZD.h>

Kernel::Kernel(){

}

Kernel::~Kernel(){

}

void Kernel::allocateStorage(User_Data *user_data){
  Grid_Data *grid_data = user_data->grid_data;
  Nesting_Order nest = nestingPsi();
  for(int gs = 0;gs < grid_data->gd_sets.size();++ gs){
    for(int ds = 0;ds < grid_data->gd_sets[gs].size();++ ds){
      Group_Dir_Set &gdset = grid_data->gd_sets[gs][ds];
      gdset.allocate(grid_data, nest);
    }
  }

  // Allocate moments variables
  int num_moments = grid_data->num_moments;
  int num_dims = 3;
  int total_dirs = user_data->directions.size();
  int num_zones = grid_data->num_zones;
  int num_groups = user_data->num_group_sets * user_data->num_groups_per_set;

  grid_data->ell = new LMat(NEST_NMD, num_dims, num_moments, total_dirs);
  grid_data->ell_plus = new LMat(NEST_DNM, num_dims, num_moments, total_dirs);

  int total_moments = grid_data->ell->total_moments;
  grid_data->phi = new SubTVec(nestingPhi(), num_groups, total_moments, num_zones);
  grid_data->phi_out = new SubTVec(nestingPhi(), num_groups, total_moments, num_zones);
}


void Kernel::evalSigmaS(Grid_Data *grid_data, int n, int g, int gp){

  int num_zones = grid_data->num_zones;
  double *sig_s = &grid_data->sig_s[0];

  for(int z = 0;z < num_zones;++ z){
    sig_s[z] = 0.0;
  }
}


// Factory to create correct kernel object
Kernel *createKernel(Nesting_Order nest, int num_dims){
  if(num_dims == 3){
    switch(nest){
    case NEST_GDZ:
      return new Kernel_3d_GDZ();
    case NEST_DGZ:
      return new Kernel_3d_DGZ();
    case NEST_ZDG:
      return new Kernel_3d_ZDG();
    case NEST_DZG:
      return new Kernel_3d_DZG();
    case NEST_ZGD:
      return new Kernel_3d_ZGD();
    case NEST_GZD:
      return new Kernel_3d_GZD();
    }
  }

  error_exit(1);
  return NULL;
}
