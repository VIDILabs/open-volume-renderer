#ifndef DIVA_CONSTANTS_H
#define DIVA_CONSTANTS_H

/* This file is going to share definitions between the host and devices */

enum struct DivaType {
  TYPE_UINT8 = 10,
  TYPE_UINT16 = 11,
  TYPE_UINT32 = 12,
  TYPE_INT8 = 20,
  TYPE_INT16 = 21,
  TYPE_INT32 = 22,
  TYPE_FLOAT = 30,
  TYPE_DOUBLE = 40
};

inline int
diva_voxel_type_size(DivaType type)
{
  switch (type) {
  case DivaType::TYPE_UINT8:
  case DivaType::TYPE_INT8: return sizeof(char);
  case DivaType::TYPE_UINT16:
  case DivaType::TYPE_INT16: return sizeof(short);
  case DivaType::TYPE_UINT32:
  case DivaType::TYPE_INT32: return sizeof(int);
  case DivaType::TYPE_FLOAT: return sizeof(float);
  case DivaType::TYPE_DOUBLE: return sizeof(double);
  default: return 0;
  }
}

struct DivaDataBuffer {};

// ... // 

enum DivaFrameBufferChannel {
  DIVA_FB_COLOR = 1,
  DIVA_FB_GRADIENT = 1 << 4,
  DIVA_FB_FLOW = 1 << 8,
};

typedef DivaType VoxelType;

enum V3D_TYPE {
  V3D_UNSIGNED_BYTE = 10,
  V3D_UNSIGNED_SHORT = 11,
  V3D_UNSIGNED_INT = 12,
  V3D_BYTE = 20,
  V3D_SHORT = 21,
  V3D_INT = 22,
  V3D_FLOAT = 30,
  V3D_DOUBLE = 40,
  V3D_VOID,
};

#endif // DIVA_CONSTANTS_H
