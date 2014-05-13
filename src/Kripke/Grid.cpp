/*--------------------------------------------------------------------------
 * Utility functions for the Grid_Data structure.
 *--------------------------------------------------------------------------*/

#include <Kripke/Grid.h>
#include <Kripke/SubTVec.h>
#include <Kripke/LMat.h>
#include <Kripke/Comm.h>
#include <Kripke/Input_Variables.h>

#include <cmath>
#include <sstream>

Group_Dir_Set::Group_Dir_Set() :
  num_groups(0),
  num_directions(0),
  group0(0),
  direction0(0),
  directions(NULL),
  psi(NULL),
  rhs(NULL),
  sigt(NULL),
  psi_lf(NULL),
  psi_fr(NULL),
  psi_bo(NULL)
{
}
Group_Dir_Set::~Group_Dir_Set(){
  delete psi;
  delete rhs;
  delete sigt;

  delete psi_lf;
  delete psi_fr;
  delete psi_bo;
}


void Group_Dir_Set::allocate(Grid_Data *grid_data, Nesting_Order nest){
  delete psi;
  psi = new SubTVec(nest,
      num_groups, num_directions, grid_data->num_zones);

  delete rhs;
  rhs = new SubTVec(nest,
      num_groups, num_directions, grid_data->num_zones);

  // allocate sigt  1xGxZ if groups come before zones
  delete sigt;
  if(nest == NEST_GDZ || nest ==  NEST_DGZ || nest == NEST_GZD){
    sigt = new SubTVec(NEST_DGZ,
      num_groups, 1, grid_data->num_zones);
  }
  // otherwise, 1xZxG
  else{
    sigt = new SubTVec(NEST_DZG,
      num_groups, 1, grid_data->num_zones);
  }

  // Allocate sweep boundary data
  int local_imax = grid_data->nzones[0];
  int local_jmax = grid_data->nzones[1];
  int local_kmax = grid_data->nzones[2];
  psi_lf = new SubTVec(nest, num_groups, num_directions,
                    (local_imax+1)*local_jmax*local_kmax);
  psi_fr = new SubTVec(nest, num_groups, num_directions,
                    local_imax*(local_jmax+1)*local_kmax);
  psi_bo = new SubTVec(nest, num_groups, num_directions,
                    local_imax*local_jmax*(local_kmax+1));
}

void Group_Dir_Set::randomizeData(void){
  psi->randomizeData();
  rhs->randomizeData();
  sigt->randomizeData();
  psi_lf->randomizeData();
  psi_fr->randomizeData();
  psi_bo->randomizeData();
}

void Group_Dir_Set::copy(Group_Dir_Set const &b){
  psi->copy(*b.psi);
  rhs->copy(*b.rhs);
  sigt->copy(*b.sigt);
  psi_lf->copy(*b.psi_lf);
  psi_fr->copy(*b.psi_fr);
  psi_bo->copy(*b.psi_bo);
}

bool Group_Dir_Set::compare(int gs, int ds, Group_Dir_Set const &b, double tol, bool verbose){
  std::stringstream namess;
  namess << "gdset[" << gs << "][" << ds << "]";
  std::string name = namess.str();

  bool is_diff = false;
  is_diff |= psi->compare(name+".psi", *b.psi, tol, verbose);
  is_diff |= rhs->compare(name+".rhs", *b.rhs, tol, verbose);
  is_diff |= sigt->compare(name+".sigt", *b.sigt, tol, verbose);
  is_diff |= psi_lf->compare(name+".psi_lf", *b.psi_lf, tol, verbose);
  is_diff |= psi_fr->compare(name+".psi_fr", *b.psi_fr, tol, verbose);
  is_diff |= psi_bo->compare(name+".psi_bo", *b.psi_bo, tol, verbose);

  return is_diff;
}


/*--------------------------------------------------------------------------
 * GenGrid : Creates a new Grid_Data structure and allocates
 *                 memory for its data.
 *
 * Currently, the spatial grid is calculated so that cells are a uniform
 * length = (xmax - xmin) / nx
 * in each spatial direction.
 *
 *--------------------------------------------------------------------------*/

Grid_Data::Grid_Data(Input_Variables *input_vars, Directions *directions)
{
  int npx = input_vars->npx;
  int npy = input_vars->npy;
  int npz = input_vars->npz;
  int nx_g = input_vars->nx;
  int ny_g = input_vars->ny;
  int nz_g = input_vars->nz;

  /* Compute the local coordinates in the processor decomposition */
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int isub_ref = myid % npx;
  int jsub_ref = ((myid - isub_ref) / npx) % npy;
  int ksub_ref = (myid - isub_ref - npx*jsub_ref) / (npx * npy);

  /* Compute the processor neighbor array assuming a lexigraphic ordering */
  if(isub_ref == 0){
    mynbr[0][0] = -1;
  }
  else {
    mynbr[0][0] = myid - 1;
  }

  if(isub_ref == npx-1){
    mynbr[0][1] = -1;
  }
  else {
    mynbr[0][1] = myid + 1;
  }

  if(jsub_ref == 0){
    mynbr[1][0] = -1;
  }
  else {
    mynbr[1][0] = myid - npx;
  }

  if(jsub_ref == npy-1){
    mynbr[1][1] = -1;
  }
  else {
    mynbr[1][1] = myid + npx;
  }

  if(ksub_ref == 0){
    mynbr[2][0] = -1;
  }
  else {
    mynbr[2][0] = myid - npx * npy;
  }

  if(ksub_ref == npz-1){
    mynbr[2][1] = -1;
  }
  else {
    mynbr[2][1] = myid + npx * npy;
  }
  
  computeGrid(0, npx, nx_g, isub_ref, 0.0, 1.0);
  computeGrid(1, npy, ny_g, jsub_ref, 0.0, 1.0);
  computeGrid(2, npz, nz_g, ksub_ref, 0.0, 1.0);
  num_zones = nzones[0]*nzones[1]*nzones[2];

  num_moments = 2;

  sig_s.resize(num_zones, 0.0);
}

Grid_Data::~Grid_Data(){
  delete phi;
  delete phi_out;
  delete ell;
  delete ell_plus;
}

void Grid_Data::randomizeData(void){
  for(int d = 0;d < 3;++ d){
    for(int i = 0;i < deltas[d].size();++ i){
      deltas[d][i] = drand48();
    }
  }

  for(int i = 0;i < volume.size();++ i){
    volume[i] = drand48();
  }

  for(int gs = 0;gs < gd_sets.size();++ gs){
    for(int ds = 0;ds < gd_sets[gs].size();++ ds){
      gd_sets[gs][ds].randomizeData();
    }
  }

  phi->randomizeData();
  phi_out->randomizeData();
  ell->randomizeData();
  ell_plus->randomizeData();

  for(int i = 0;i < sig_s.size();++ i){
    sig_s[i] = drand48();
  }
}

void Grid_Data::copy(Grid_Data const &b){
  for(int d = 0;d < 3;++ d){
    deltas[d] = b.deltas[d];
  }
  volume = b.volume;

  for(int gs = 0;gs < gd_sets.size();++ gs){
    for(int ds = 0;ds < gd_sets[gs].size();++ ds){
      gd_sets[gs][ds].copy(b.gd_sets[gs][ds]);
    }
  }
  phi->copy(*b.phi);
  phi_out->copy(*b.phi_out);
  ell->copy(*b.ell);
  ell_plus->copy(*b.ell_plus);

  sig_s = b.sig_s;
}

bool Grid_Data::compare(Grid_Data const &b, double tol, bool verbose){
  bool is_diff = false;
  is_diff |= compareVector("deltas[0]", deltas[0], b.deltas[0], tol, verbose);
  is_diff |= compareVector("deltas[1]", deltas[1], b.deltas[1], tol, verbose);
  is_diff |= compareVector("deltas[2]", deltas[2], b.deltas[2], tol, verbose);

  is_diff |= compareVector("volume", volume, b.volume, tol, verbose);

  for(int gs = 0;gs < gd_sets.size();++ gs){
    for(int ds = 0;ds < gd_sets[gs].size();++ ds){
      is_diff |= gd_sets[gs][ds].compare(
          gs, ds, b.gd_sets[gs][ds], tol, verbose);
    }
  }

  is_diff |= phi->compare("phi", *b.phi, tol, verbose);
  is_diff |= phi_out->compare("phi_out", *b.phi_out, tol, verbose);
  is_diff |= ell->compare("ell", *b.ell, tol, verbose);
  is_diff |= ell_plus->compare("ell_plus", *b.ell_plus, tol, verbose);

  return is_diff;
}


void Grid_Data::computeGrid(int dim, int npx, int nx_g, int isub_ref, double xmin, double xmax){
 /* Calculate unit roundoff and load into grid_data */
  double eps = 1e-32;
  double thsnd_eps = 1000.e0*(eps);
 
  // Compute subset of global zone indices
  int nx_l = nx_g / npx;
  int rem = nx_g % npx;
  int ilower, iupper;
  if(rem != 0){
    if(isub_ref < rem){
      nx_l++;
      ilower = isub_ref * nx_l;
    }
    else {
      ilower = rem + isub_ref * nx_l;
    }
  }
  else {
    ilower = isub_ref * nx_l;
  }

  iupper = ilower + nx_l - 1;

  // allocate grid deltas
  deltas[dim].resize(nx_l+2);
  
  // Compute the spatial grid 
  double dx = (xmax - xmin) / nx_g;
  double coord_lo = xmin + (ilower) * dx;
  double coord_hi = xmin + (iupper+1) * dx;
  for(int i = 0; i < nx_l+2; i++){
    deltas[dim][i] = dx;
  }
  if(std::abs(coord_lo - xmin) <= thsnd_eps*std::abs(xmin)){
    deltas[dim][0] = 0.0;
  }
  if(std::abs(coord_hi - xmax) <= thsnd_eps*std::abs(xmax)){
    deltas[dim][nx_l+1] = 0.0;
  }
  
  nzones[dim] = nx_l; 
}