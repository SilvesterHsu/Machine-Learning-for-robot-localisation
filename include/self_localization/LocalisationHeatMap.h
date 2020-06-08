#ifndef MAPPED_REGION_H
#define MAPPED_REGION_H
#include <math.h>
#include <stdio.h>
#include <vector>
#include <cstdio>
#include <stdlib.h>
#include <cstring>
#include "Eigen/Dense"
#include <iostream>


using std::cout;
using std::endl;
namespace HeatMap {

class Voxel
{
public:
  Voxel() {}
  float val=0;
  static Voxel* CreateVoxel(){return new Voxel();}
};


typedef  std::pair<Voxel*, Eigen::Vector3i> IndexedVoxel; //contains metadata about the voxel

typedef std::vector< IndexedVoxel > VoxelVector;

class LocalisationHeatMap
{  

public:

  LocalisationHeatMap(float resolution, const double map_size_x, const double map_size_y, const double map_size_z);

  bool getNeighborsByRadius(const double &radius, const Eigen::Vector3f &p, VoxelVector &nearby_voxels,bool OnlyInitialized=true);

  Eigen::Vector3f GetVoxelCenter(const Eigen::Vector3i &idx);

  Eigen::Vector3f GetVoxelCenter(const IndexedVoxel &voxel);

  double DistanceP2VCenter(const Eigen::Vector3f &p, const IndexedVoxel &voxel);

  void PrintGrid();

  void getIndexForPoint(const Eigen::Vector3f &p, Eigen::Vector3i &idx);

  static void VoxelFilter(VoxelVector &vek, bool remove_initialized_voxels=false);

  void UpdateHeatmapNormal(const Eigen::Affine3d &Tupdsrc, float sigma_update = 10);

  void UpdateHeatmapNormal(const Eigen::Vector3f &Tupdsrc, float sigma_update = 10);

  void InitializeNeighborsByRadius(const double &radius, const Eigen::Vector3f &p);

  void InitializeNeighborsByRadius(const double &radius, const Eigen::Vector3f &p, VoxelVector &nearby_voxels);


private:

  float sizeXmeters, sizeYmeters, sizeZmeters;
  float resolution_;
  int sizeX, sizeY, sizeZ;
  float half_voxel;
  float half_sizeXmeters, half_sizeYmeters, half_sizeZmeters;
  Voxel ****dataArray;
  unsigned int n_updates_ = 0;
  float sigma_update_;  //applicable when using gausian accuracy decay


};


}
#endif // MAPPED_REGION_H
