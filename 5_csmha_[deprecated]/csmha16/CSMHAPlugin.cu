/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CSMHAPlugin.h"

#define ALIGNSIZE  1024
#define ALIGNED(x) (((x) + ALIGNSIZE - 1) / ALIGNSIZE * ALIGNSIZE)

struct __align__(16) half8
{
  half a;
  half b;
  half c;
  half d;
  half e;
  half f;
  half g;
  half h;
};

namespace nvinfer1
{
// class CSMHAPlugin
CSMHAPlugin::CSMHAPlugin(const std::string &name, 
                            nvinfer1::Weights qweight, nvinfer1::Weights qbias, 
                            nvinfer1::Weights kweight, nvinfer1::Weights kbias, 
                            nvinfer1::Weights vweight, nvinfer1::Weights vbias, 
                            nvinfer1::Weights oweight, nvinfer1::Weights obias, 
                            nvinfer1::Weights qkbias, nvinfer1::Weights qcbias, 
                            int k, int n):
    name_(name), nK_(k), nN_(n)
{
    WHERE_AM_I()
    
    {
        // Load Q weight and bias
        {
            assert(qweight.type == DataType::kFLOAT);
            assert(qweight.values != nullptr);
            assert(qweight.count == k * n);

            qweight_.type  = DataType::kFLOAT;
            qweight_.count = qweight.count;
            size_t size    = sizeof(float) * qweight.count;
            qweight_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(qweight_.values)), qweight.values, size);

            qweight16_.type   = DataType::kHALF;
            qweight16_.count  = nK_ * nN_;
            size            = sizeof(half) * nK_ * nN_;
            qweight16_.values = malloc(size);
            for(int i = 0; i < qweight16_.count; i++)
                ((half*)qweight16_.values)[i] = (half)(((float*)qweight_.values)[i]);
        }
        {
            assert(qbias.type == DataType::kFLOAT);
            assert(qbias.values != nullptr);
            assert(qbias.count == n);

            qbias_.type  = DataType::kFLOAT;
            qbias_.count = qbias.count;
            size_t size    = sizeof(float) * qbias.count;
            qbias_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(qbias_.values)), qbias.values, size);

            qbias16_.type   = DataType::kHALF;
            qbias16_.count  = nN_;
            size          = sizeof(half) * nN_;
            qbias16_.values = malloc(size);
            for(int i = 0; i < qbias16_.count; i++)
                ((half*)qbias16_.values)[i] = (half)(((float*)qbias_.values)[i]);
        }
    }

    {
        // Load K weight and bias
        {
            assert(kweight.type == DataType::kFLOAT);
            assert(kweight.values != nullptr);
            assert(kweight.count == k * n);

            kweight_.type  = DataType::kFLOAT;
            kweight_.count = kweight.count;
            size_t size    = sizeof(float) * kweight.count;
            kweight_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(kweight_.values)), kweight.values, size);

            kweight16_.type   = DataType::kHALF;
            kweight16_.count  = nK_ * nN_;
            size            = sizeof(half) * nK_ * nN_;
            kweight16_.values = malloc(size);
            for(int i = 0; i < kweight16_.count; i++)
                ((half*)kweight16_.values)[i] = (half)(((float*)kweight_.values)[i]);
        }
        {
            assert(kbias.type == DataType::kFLOAT);
            assert(kbias.values != nullptr);
            assert(kbias.count == n);

            kbias_.type  = DataType::kFLOAT;
            kbias_.count = kbias.count;
            size_t size    = sizeof(float) * kbias.count;
            kbias_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(kbias_.values)), kbias.values, size);

            kbias16_.type   = DataType::kHALF;
            kbias16_.count  = nN_;
            size          = sizeof(half) * nN_;
            kbias16_.values = malloc(size);
            for(int i = 0; i < kbias16_.count; i++)
                ((half*)kbias16_.values)[i] = (half)(((float*)kbias_.values)[i]);
        }
    }

    {
        // Load V weight and bias
        {
            assert(vweight.type == DataType::kFLOAT);
            assert(vweight.values != nullptr);
            assert(vweight.count == k * n);

            vweight_.type  = DataType::kFLOAT;
            vweight_.count = vweight.count;
            size_t size    = sizeof(float) * vweight.count;
            vweight_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(vweight_.values)), vweight.values, size);

            vweight16_.type   = DataType::kHALF;
            vweight16_.count  = nK_ * nN_;
            size            = sizeof(half) * nK_ * nN_;
            vweight16_.values = malloc(size);
            for(int i = 0; i < vweight16_.count; i++)
                ((half*)vweight16_.values)[i] = (half)(((float*)vweight_.values)[i]);
        }
        {
            assert(vbias.type == DataType::kFLOAT);
            assert(vbias.values != nullptr);
            assert(vbias.count == n);

            vbias_.type  = DataType::kFLOAT;
            vbias_.count = vbias.count;
            size_t size    = sizeof(float) * vbias.count;
            vbias_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(vbias_.values)), vbias.values, size);

            vbias16_.type   = DataType::kHALF;
            vbias16_.count  = nN_;
            size          = sizeof(half) * nN_;
            vbias16_.values = malloc(size);
            for(int i = 0; i < vbias16_.count; i++)
                ((half*)vbias16_.values)[i] = (half)(((float*)vbias_.values)[i]);
        }
    }

    {
        // Load O weight and bias
        {
            assert(oweight.type == DataType::kFLOAT);
            assert(oweight.values != nullptr);
            assert(oweight.count == k * n);

            oweight_.type  = DataType::kFLOAT;
            oweight_.count = oweight.count;
            size_t size    = sizeof(float) * oweight.count;
            oweight_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(oweight_.values)), oweight.values, size);

            oweight16_.type   = DataType::kHALF;
            oweight16_.count  = nK_ * nN_;
            size            = sizeof(half) * nK_ * nN_;
            oweight16_.values = malloc(size);
            for(int i = 0; i < oweight16_.count; i++)
                ((half*)oweight16_.values)[i] = (half)(((float*)oweight_.values)[i]);
        }
        {
            assert(obias.type == DataType::kFLOAT);
            assert(obias.values != nullptr);
            assert(obias.count == n);

            obias_.type  = DataType::kFLOAT;
            obias_.count = obias.count;
            size_t size    = sizeof(float) * obias.count;
            obias_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(obias_.values)), obias.values, size);

            obias16_.type   = DataType::kHALF;
            obias16_.count  = nN_;
            size          = sizeof(half) * nN_;
            obias16_.values = malloc(size);
            for(int i = 0; i < obias16_.count; i++)
                ((half*)obias16_.values)[i] = (half)(((float*)obias_.values)[i]);
        }
    }

    {
        // Load qk and qc bias
        {
            assert(qkbias.type == DataType::kFLOAT);
            assert(qkbias.values != nullptr);
            assert(qkbias.count == n);

            qkbias_.type  = DataType::kFLOAT;
            qkbias_.count = qkbias.count;
            size_t size    = sizeof(float) * qkbias.count;
            qkbias_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(qkbias_.values)), qkbias.values, size);

            qkbias16_.type   = DataType::kHALF;
            qkbias16_.count  = nN_;
            size          = sizeof(half) * nN_;
            qkbias16_.values = malloc(size);
            for(int i = 0; i < qkbias16_.count; i++)
                ((half*)qkbias16_.values)[i] = (half)(((float*)qkbias_.values)[i]);
        }
        {
            assert(qcbias.type == DataType::kFLOAT);
            assert(qcbias.values != nullptr);
            assert(qcbias.count == n);

            qcbias_.type  = DataType::kFLOAT;
            qcbias_.count = qcbias.count;
            size_t size    = sizeof(float) * qcbias.count;
            qcbias_.values = malloc(size);
            memcpy(reinterpret_cast<char *>(const_cast<void *>(qcbias_.values)), qcbias.values, size);

            qcbias16_.type   = DataType::kHALF;
            qcbias16_.count  = nN_;
            size          = sizeof(half) * nN_;
            qcbias16_.values = malloc(size);
            for(int i = 0; i < qcbias16_.count; i++)
                ((half*)qcbias16_.values)[i] = (half)(((float*)qcbias_.values)[i]);
        }
    }

    CHECK(cublasCreate(&handle_));
}

CSMHAPlugin::CSMHAPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    WHERE_AM_I()
    const char *data   = reinterpret_cast<const char *>(buffer);
    size_t      offset = 0;
    memcpy(&nK_, data + offset, sizeof(nK_));
    offset += sizeof(nK_);
    memcpy(&nN_, data + offset, sizeof(nN_));
    offset += sizeof(nN_);


    size_t size;

    {
        qweight_.type   = DataType::kFLOAT;
        qweight_.count  = nK_ * nN_;
        size            = sizeof(float) * nK_ * nN_;
        qweight_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(qweight_.values)), data + offset, size);
        offset += size;

        qbias_.type   = DataType::kFLOAT;
        qbias_.count  = nN_;
        size          = sizeof(float) * nN_;
        qbias_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(qbias_.values)), data + offset, size);
        offset += size;

        qweight16_.type   = DataType::kHALF;
        qweight16_.count  = nK_ * nN_;
        size            = sizeof(half) * nK_ * nN_;
        qweight16_.values = malloc(size);
        for(int i = 0; i < qweight16_.count; i++)
            ((half*)qweight16_.values)[i] = (half)(((float*)qweight_.values)[i]);

        qbias16_.type   = DataType::kHALF;
        qbias16_.count  = nN_;
        size          = sizeof(half) * nN_;
        qbias16_.values = malloc(size);
        for(int i = 0; i < qbias16_.count; i++)
            ((half*)qbias16_.values)[i] = (half)(((float*)qbias_.values)[i]);
    }

    {
        kweight_.type   = DataType::kFLOAT;
        kweight_.count  = nK_ * nN_;
        size            = sizeof(float) * nK_ * nN_;
        kweight_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(kweight_.values)), data + offset, size);
        offset += size;

        kbias_.type   = DataType::kFLOAT;
        kbias_.count  = nN_;
        size         = sizeof(float) * nN_;
        kbias_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(kbias_.values)), data + offset, size);
        offset += size;

        kweight16_.type   = DataType::kHALF;
        kweight16_.count  = nK_ * nN_;
        size            = sizeof(half) * nK_ * nN_;
        kweight16_.values = malloc(size);
        for(int i = 0; i < kweight16_.count; i++)
            ((half*)kweight16_.values)[i] = (half)(((float*)kweight_.values)[i]);

        kbias16_.type   = DataType::kHALF;
        kbias16_.count  = nN_;
        size          = sizeof(half) * nN_;
        kbias16_.values = malloc(size);
        for(int i = 0; i < kbias16_.count; i++)
            ((half*)kbias16_.values)[i] = (half)(((float*)kbias_.values)[i]);
    }

    {
        vweight_.type   = DataType::kFLOAT;
        vweight_.count  = nK_ * nN_;
        size            = sizeof(float) * nK_ * nN_;
        vweight_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(vweight_.values)), data + offset, size);
        offset += size;

        vbias_.type   = DataType::kFLOAT;
        vbias_.count  = nN_;
        size         = sizeof(float) * nN_;
        vbias_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(vbias_.values)), data + offset, size);
        offset += size;

        vweight16_.type   = DataType::kHALF;
        vweight16_.count  = nK_ * nN_;
        size            = sizeof(half) * nK_ * nN_;
        vweight16_.values = malloc(size);
        for(int i = 0; i < vweight16_.count; i++)
            ((half*)vweight16_.values)[i] = (half)(((float*)vweight_.values)[i]);

        vbias16_.type   = DataType::kHALF;
        vbias16_.count  = nN_;
        size          = sizeof(half) * nN_;
        vbias16_.values = malloc(size);
        for(int i = 0; i < vbias16_.count; i++)
            ((half*)vbias16_.values)[i] = (half)(((float*)vbias_.values)[i]);
    }

    {
        oweight_.type   = DataType::kFLOAT;
        oweight_.count  = nK_ * nN_;
        size            = sizeof(float) * nK_ * nN_;
        oweight_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(oweight_.values)), data + offset, size);
        offset += size;

        obias_.type   = DataType::kFLOAT;
        obias_.count  = nN_;
        size         = sizeof(float) * nN_;
        obias_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(obias_.values)), data + offset, size);
        offset += size;

        oweight16_.type   = DataType::kHALF;
        oweight16_.count  = nK_ * nN_;
        size            = sizeof(half) * nK_ * nN_;
        oweight16_.values = malloc(size);
        for(int i = 0; i < oweight16_.count; i++)
            ((half*)oweight16_.values)[i] = (half)(((float*)oweight_.values)[i]);

        obias16_.type   = DataType::kHALF;
        obias16_.count  = nN_;
        size          = sizeof(half) * nN_;
        obias16_.values = malloc(size);
        for(int i = 0; i < obias16_.count; i++)
            ((half*)obias16_.values)[i] = (half)(((float*)obias_.values)[i]);
    }

    {
        qkbias_.type   = DataType::kFLOAT;
        qkbias_.count  = nN_;
        size         = sizeof(float) * nN_;
        qkbias_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(qkbias_.values)), data + offset, size);
        offset += size;

        qcbias_.type   = DataType::kFLOAT;
        qcbias_.count  = nN_;
        size         = sizeof(float) * nN_;
        qcbias_.values = malloc(size);
        memcpy(reinterpret_cast<char *>(const_cast<void *>(qcbias_.values)), data + offset, size);
        offset += size;

        qkbias16_.type   = DataType::kHALF;
        qkbias16_.count  = nN_;
        size          = sizeof(half) * nN_;
        qkbias16_.values = malloc(size);
        for(int i = 0; i < qkbias16_.count; i++)
            ((half*)qkbias16_.values)[i] = (half)(((float*)qkbias_.values)[i]);

        qcbias16_.type   = DataType::kHALF;
        qcbias16_.count  = nN_;
        size          = sizeof(half) * nN_;
        qcbias16_.values = malloc(size);
        for(int i = 0; i < qcbias16_.count; i++)
            ((half*)qcbias16_.values)[i] = (half)(((float*)qcbias_.values)[i]);
    }

    CHECK(cublasCreate(&handle_));
}

CSMHAPlugin::~CSMHAPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *CSMHAPlugin::clone() const noexcept
{
    WHERE_AM_I()
    CSMHAPlugin *p = new CSMHAPlugin(name_, 
                                        qweight_, qbias_, 
                                        kweight_, kbias_, 
                                        vweight_, vbias_, 
                                        oweight_, obias_, 
                                        qkbias_, qcbias_, 
                                        nK_, nN_);
    p->setPluginNamespace(namespace_.c_str());
    p->pGPUQWeight_ = this->pGPUQWeight_;
    p->pGPUQBias_ = this->pGPUQBias_;
    p->pGPUKWeight_ = this->pGPUKWeight_;
    p->pGPUKBias_ = this->pGPUKBias_;
    p->pGPUVWeight_ = this->pGPUVWeight_;
    p->pGPUVBias_ = this->pGPUVBias_;
    p->pGPUOWeight_ = this->pGPUOWeight_;
    p->pGPUOBias_ = this->pGPUOBias_;
    p->pGPUQKBias_ = this->pGPUQKBias_;
    p->pGPUQCBias_ = this->pGPUQCBias_;

    p->pGPUQWeight16_ = this->pGPUQWeight16_;
    p->pGPUQBias16_ = this->pGPUQBias16_;
    p->pGPUKWeight16_ = this->pGPUKWeight16_;
    p->pGPUKBias16_ = this->pGPUKBias16_;
    p->pGPUVWeight16_ = this->pGPUVWeight16_;
    p->pGPUVBias16_ = this->pGPUVBias16_;
    p->pGPUOWeight16_ = this->pGPUOWeight16_;
    p->pGPUOBias16_ = this->pGPUOBias16_;
    p->pGPUQKBias16_ = this->pGPUQKBias16_;
    p->pGPUQCBias16_ = this->pGPUQCBias16_;
    return p;
}

int32_t CSMHAPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I()
    return 1;
}

DataType CSMHAPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I()
    return inputTypes[0];
}

DimsExprs CSMHAPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I()
    // DimsExprs ret {inputs[0]};
    // ret.d[inputs[0].nbDims - 1] = exprBuilder.constant(nN_);
    // return ret;
    return inputs[0];
}

bool CSMHAPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I()
    switch (pos)
    {
        case 0: return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        case 1: return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
        case 2: return inOut[2].type == DataType::kFLOAT && inOut[2].format == TensorFormat::kLINEAR;
        case 3: return inOut[3].type == DataType::kFLOAT && inOut[3].format == TensorFormat::kLINEAR;
        default: return false;
    }
    return false;
}

void CSMHAPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
}

size_t CSMHAPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I()

    return ALIGNED(
        13 * sizeof(float) * 2 * 192 * 192
    );

    return 0;
}


template<typename T, int BX, int TX>
__global__ void addBiasByCopyKernel(T *inOut, T *bias)
{
    const int spaceIndex = blockIdx.y * BX * TX + blockIdx.x * TX + threadIdx.x * 8;
    const int dataIndex = threadIdx.x * 8;

    reinterpret_cast<half8*>(inOut+spaceIndex)[0] = reinterpret_cast<half8*>(bias+dataIndex)[0];
}

template<typename T, int THREAD>
__global__ void addBiasKernel(T *pInput, T *pOutput, T *bias)
{
    const int tx = threadIdx.x, index = blockIdx.x * THREAD + threadIdx.x;
    T _x = pInput[index], _b = bias[tx];
    pOutput[index] = _x + _b;
}

template<typename T, int TY, int TX>
__global__ void faster_transposeQKernel(T *pInput, T *pOutput)
{
    const int by = blockIdx.y, bx = blockIdx.x;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int BX = gridDim.x;

    const int inIndex = by*(BX*TY*TX)+bx*(TY*TX)+ty*TX+tx*8;
    const int ouIndex = by*(TY*BX*TX)+ty*(BX*TX)+bx*TX+tx*8;

    reinterpret_cast<half8*>(pOutput+ouIndex)[0] = reinterpret_cast<half8*>(pInput+inIndex)[0];
}

template<typename T, int TY, int TX>
__global__ void faster_transposeIQKernel(T *pInput, T *pOutput)
{
    const int by = blockIdx.y, bx = blockIdx.x;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int BX = gridDim.x;

    const int inIndex = by*(TY*BX*TX)+ty*(BX*TX)+bx*TX+tx*8;
    const int ouIndex = by*(BX*TY*TX)+bx*(TY*TX)+ty*TX+tx*8;

    reinterpret_cast<half8*>(pOutput+ouIndex)[0] = reinterpret_cast<half8*>(pInput+inIndex)[0];
}

// (B, X, Y) -> (B, Y, X)
// https://github.com/Oneflow-Inc/oneflow/blob/f0e9d38b2ba4ac535fd6de5dbeca4e3d2051de23/oneflow/core/ep/cuda/primitive/permute.cu#L57
template<typename T, size_t num_dims, size_t tile_size>
__global__ void faster_transposeKKernel(void* src_ptr, void* dst_ptr, int rows, int cols, int num_tile_rows, int num_tile_cols, int32_t block_nums)
{
    const int src_rows = rows;
    const int src_cols = cols;
    const int dst_rows = cols;
    const int dst_cols = rows;

    __shared__ T tile[tile_size][tile_size + 1];

    const T* src = reinterpret_cast<const T*>(src_ptr);
    T* dst = reinterpret_cast<T*>(dst_ptr);

    int batch_num_tile = num_tile_rows * num_tile_cols;
    for (int i = blockIdx.x, step = gridDim.x; i < block_nums; i += step) 
    {
        const int batch_index = i / batch_num_tile;
        const int tile_index = i - batch_index * batch_num_tile;

        const int tile_row_index = tile_index / num_tile_cols;
        const int tile_col_index = tile_index - tile_row_index * num_tile_cols;

        const int offset = batch_index * src_rows * src_cols;
        {
            int col_in_tile = threadIdx.x;
            int col_in_matrix = tile_col_index * tile_size + threadIdx.x;
#pragma unroll
            for (int row_in_tile = threadIdx.y; row_in_tile < tile_size; row_in_tile += 8) 
            {
                int row_in_matrix = row_in_tile + tile_row_index * tile_size;
                if (col_in_matrix < src_cols && row_in_matrix < src_rows) 
                {
                    tile[row_in_tile][col_in_tile] = src[offset + row_in_matrix * src_cols + col_in_matrix];
                }
            }
        }
        __syncthreads();
        {
            int col_in_tile = threadIdx.x;
            int col_in_matrix = tile_row_index * tile_size + threadIdx.x;
#pragma unroll
            for (int row_in_tile = threadIdx.y; row_in_tile < tile_size; row_in_tile += 8) 
            {
                int row_in_matrix = row_in_tile + tile_col_index * tile_size;
                if (col_in_matrix < dst_cols && row_in_matrix < dst_rows) 
                {
                    dst[offset + row_in_matrix * dst_cols + col_in_matrix] = tile[col_in_tile][row_in_tile];
                }
            }
        }
        __syncthreads();
    }
}

template<typename T>
__global__ void concat1Kernel(T *pInput, T *pOutput, int m)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int ouIndex = bx * 2 * m + tx;
    
    if (tx == 0)
        pOutput[ouIndex] = 0;
    else
        pOutput[ouIndex] = pInput[ouIndex-bx-1];
}

template<typename T>
__global__ void slice1Kernel(T *pInput, T *pOutput, int m)
{
    const int by = blockIdx.y, bx = blockIdx.x, tx = threadIdx.x;

    const int ouIndex = by*(2*m*m-m) + bx*m   + tx;
    const int inIndex = by*(2*m*m)   + bx*m+m + tx;

    pOutput[ouIndex] = pInput[inIndex];
}

template<typename T>
__global__ void slice2Kernel(T *pInput, T *pOutput, int m)
{
    const int bx = blockIdx.x, tx = threadIdx.x;

    const int ouIndex = bx*m + tx;
    const int inIndex = bx*(2*m-1) + tx;

    pOutput[ouIndex] = pInput[inIndex];
}

template<typename T>
__global__ void addScaleKernel(T *p1Input, T *p2Input, int m)
{
    const int bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    const int inOuIndex = bx*m*m + ty*m + tx;

    p2Input[inOuIndex] = (p1Input[inOuIndex] + p2Input[inOuIndex]) / 9.797959327697754;
}

template<typename T>
__global__ void scaleMaskKernel(T *p1Input, T *p2Input, float *pMask, T *pOutput, int m)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int inOuIndex = bx*m + tx;

    if(pMask[tx] == 0)
        pOutput[inOuIndex] = -3.4028234663852886e+38;
    else
    {
        pOutput[inOuIndex] = p1Input[inOuIndex] + p2Input[inOuIndex];
        pOutput[inOuIndex] = pOutput[inOuIndex] / (half)9.797959327697754;
    }
}

template<typename T>
__global__ void rowwiseMaxKernel(const T *pInput, T *pOutput, int m)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int index = bx*m + tx;

    T _x = -3.4028234663852886e+38;
    if(tx<m)
        _x = pInput[index];

    typedef cub::BlockReduce<T, 144>              BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T                                            max  = BlockReduce(temp).Reduce(_x, cub::Max());

    if(tx == 0)
    {
        pOutput[bx] = max;
    }
}

template<typename T>
__global__ void expKernel(const T *pInput, const T *pMax, T *pOutput, int m)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int index = bx*m + tx;

    T _x = pInput[index];
    T _m = pMax[bx];

    pOutput[index] = expf(_x - _m);
}

template<typename T>
__global__ void rowwiseSumKernel(const T *pInput, T *pOutput, int m)
{
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int index = bx*m + tx;

    T _x = 0.0f;
    if(tx<m)
        _x = pInput[index];

    typedef cub::BlockReduce<T, 144>              BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T                                            max  = BlockReduce(temp).Reduce(_x, cub::Sum());

    if(tx == 0)
    {
        pOutput[bx] = max;
    }
}

template<typename T>
__global__ void divKernel(const T *pInput, const T *pSum, T *pOutput, int m)
{
    const int tx = threadIdx.x, bx = blockIdx.x;
    const int index = bx*m + tx;

    T _x = pInput[index];
    T _s = pSum[bx];

    pOutput[index] = _x / _s;
}

__global__ void dataToFP16Kernel(float *inFP32, half *outFP16)
{
    const int index = blockIdx.x * 192 + threadIdx.x;
    outFP16[index] = (half)(inFP32[index]);
}

__global__ void dataToFP32Kernel(half *inFP16, float *outFP32)
{
    const int index = blockIdx.x * 192 + threadIdx.x;
    outFP32[index] = (float)(inFP16[index]);
}


int32_t CSMHAPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I()

    if (nK_ == 192 && nN_ == 192 && 
        inputDesc[0].dims.nbDims == 3 && inputDesc[0].dims.d[0] == 1 && inputDesc[0].dims.d[2] == 192 &&
        inputDesc[1].dims.nbDims == 4 && inputDesc[1].dims.d[0] == 1 && inputDesc[1].dims.d[1] == 2 && inputDesc[1].dims.d[2] == 96 &&
        inputDesc[2].dims.nbDims == 3 && inputDesc[2].dims.d[0] == 1 && inputDesc[2].dims.d[1] == 1
    )
    {

        const int   m = inputDesc[0].dims.d[1];
        const int   mm = inputDesc[1].dims.d[3];
        const float alpha1 = 1.0f, beta0 = 0.0f, beta1 = 1.0f;
        const half  alpha1_16 = 1.0f, beta0_16 = 0.0f, beta1_16 = 1.0f;
        const int   G = sizeof(float) * 2 * 192 * 192;

        dataToFP16Kernel<<<m, 192, 0, stream>>>((float *)inputs[0], (half *)(workspace+11*G));
        dataToFP16Kernel<<<mm, 192, 0, stream>>>((float *)inputs[1], (half *)(workspace+12*G));

        // Q
        (addBiasByCopyKernel<half, 1, 192>)<<<dim3(1,m,1), dim3(24,1,1), 0, stream>>>((half *)(workspace+0*G), pGPUQBias16_);
        CHECK(cublasSgemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, nN_, m, nK_, &alpha1, pGPUQWeight16_, CUDA_R_16F, nN_, (half *)(workspace+11*G), CUDA_R_16F, nK_, &beta1, (half *)(workspace+0*G), CUDA_R_16F, nN_));
        // 0

        // K
        (addBiasByCopyKernel<half, 1, 192>)<<<dim3(1,m,1), dim3(24,1,1), 0, stream>>>((half *)(workspace+1*G), pGPUKBias16_);
        CHECK(cublasSgemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, nN_, m, nK_, &alpha1, pGPUKWeight16_, CUDA_R_16F, nN_, (half *)(workspace+11*G), CUDA_R_16F, nK_, &beta1, (half *)(workspace+1*G), CUDA_R_16F, nN_));
        // 0,1

        // V
        (addBiasByCopyKernel<half, 1, 192>)<<<dim3(1,m,1), dim3(24,1,1), 0, stream>>>((half *)(workspace+2*G), pGPUVBias16_);
        CHECK(cublasSgemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, nN_, m, nK_, &alpha1, pGPUVWeight16_, CUDA_R_16F, nN_, (half *)(workspace+11*G), CUDA_R_16F, nK_, &beta1, (half *)(workspace+2*G), CUDA_R_16F, nN_));
        // 0,1,2

        // QK QC
        (addBiasKernel<half, 192>)<<<m, 192, 0, stream>>>((half *)(workspace+0*G), (half *)(workspace+3*G), pGPUQKBias16_);
        (addBiasKernel<half, 192>)<<<m, 192, 0, stream>>>((half *)(workspace+0*G), (half *)(workspace+4*G), pGPUQCBias16_);
        (faster_transposeQKernel<half, 2, 96>)<<<dim3(m,1,1), dim3(12,2,1), 0, stream>>>((half *)(workspace+3*G), (half *)(workspace+5*G));
        (faster_transposeQKernel<half, 2, 96>)<<<dim3(m,1,1), dim3(12,2,1), 0, stream>>>((half *)(workspace+4*G), (half *)(workspace+6*G));
        int tile_size = 16;
        int rows = m;
        int cols = 192;
        int num_tile_rows = (rows + tile_size - 1) / tile_size;
        int num_tile_cols = (cols + tile_size - 1) / tile_size;
        int32_t block_nums = 1 * num_tile_rows * num_tile_cols;
        (faster_transposeKKernel<half, 3, 16>)<<<block_nums, dim3(16, 8), 0, stream>>>((half *)(workspace+1*G), (half *)(workspace+7*G), rows, cols, num_tile_rows, num_tile_cols, block_nums);
        cublasHgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 96, &alpha1_16, (half *)(workspace+7*G), m, 96*m, (half *)(workspace+5*G), 96, 96*m, &beta0_16, (half *)(workspace+8*G), m, m*m, 2);
        cublasHgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, 2*m-1, m, 96, &alpha1_16, (half *)(workspace+12*G), 2*m-1, 96*(2*m-1), (half *)(workspace+6*G), 96, 96*m, &beta0_16, (half *)(workspace+9*G), 2*m-1, m*(2*m-1), 2);

        // Slice
        (concat1Kernel<half>)<<<2*m,2*m,0,stream>>>((half *)(workspace+9*G),(half *)(workspace+0*G),m);
        (slice1Kernel<half>)<<<dim3(2*m-1,2,1), m, 0, stream>>>((half *)(workspace+0*G),(half *)(workspace+1*G),m);
        (slice2Kernel<half>)<<<2*m, m, 0, stream>>>((half *)(workspace+1*G),(half *)(workspace+3*G),m);

        // AddMaskSoftmax
        (scaleMaskKernel<half>)<<<2*m, m, 0, stream>>>((half *)(workspace+8*G),(half *)(workspace+3*G),(float *)(inputs[2]),(half *)(workspace+3*G),m);
        (rowwiseMaxKernel<half>)<<<2*m, 144, 0, stream>>>((half *)(workspace+3*G),(half *)(workspace+1*G),m);
        (expKernel<half>)<<<2*m, m, 0, stream>>>((half *)(workspace+3*G),(half *)(workspace+1*G),(half *)(workspace+3*G),m);
        (rowwiseSumKernel<half>)<<<2*m, 144, 0, stream>>>((half *)(workspace+3*G),(half *)(workspace+1*G),m);
        (divKernel<half>)<<<2*m, m, 0, stream>>>((half *)(workspace+3*G),(half *)(workspace+1*G),(half *)(workspace+3*G),m);

        // QKV
        (faster_transposeQKernel<half, 2, 96>)<<<dim3(m,1,1), dim3(12,2,1), 0, stream>>>((half *)(workspace+2*G), (half *)(workspace+4*G));
        cublasHgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, 96, m, m, &alpha1_16, (half *)(workspace+4*G), 96, 96*m, (half *)(workspace+3*G), m, m*m, &beta0_16, (half *)(workspace+5*G), 96, m*96, 2);
        (faster_transposeIQKernel<half, 2, 96>)<<<dim3(m,1,1), dim3(12,2,1), 0, stream>>>((half *)(workspace+5*G), (half *)(workspace+6*G));

        // O
        (addBiasByCopyKernel<half, 1, 192>)<<<dim3(1,m,1), dim3(24,1,1), 0, stream>>>((half *)(workspace+8*G), pGPUOBias16_);
        CHECK(cublasSgemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, nN_, m, nK_, &alpha1, pGPUOWeight16_, CUDA_R_16F, nN_, (half *)(workspace+6*G), CUDA_R_16F, nK_, &beta1, (half *)(workspace+8*G), CUDA_R_16F, nN_));
   
        dataToFP32Kernel<<<m, 192, 0, stream>>>((half *)(workspace+8*G), (float *)(outputs[0]));
   
   
   
    }
    else
    {
        printf("[Kernel] [CSMHA] [CSMHAPlugin] [enqueue] only support input with (1,1,192) (1,2,96,1) (1,1,1), weight with (192,192) and bias with (192)\n");
        printf("[Kernel] [CSMHA] [CSMHAPlugin] [enqueue] but %d(%d,%d,%d) %d(%d,%d,%d,%d) %d(%d,%d,%d)\n",
            inputDesc[0].dims.nbDims,inputDesc[0].dims.d[0],inputDesc[0].dims.d[1],inputDesc[0].dims.d[2],
            inputDesc[1].dims.nbDims,inputDesc[1].dims.d[0],inputDesc[1].dims.d[1],inputDesc[1].dims.d[2],inputDesc[1].dims.d[3],
            inputDesc[2].dims.nbDims,inputDesc[2].dims.d[0],inputDesc[2].dims.d[1],inputDesc[2].dims.d[2]
        );
    }


    return 0;
}

int32_t CSMHAPlugin::initialize() noexcept
{
    WHERE_AM_I()

    size_t size;

    {
        size = sizeof(float) * qweight_.count;
        CHECK(cudaMalloc((void **)&pGPUQWeight_, size));
        CHECK(cudaMemcpy(pGPUQWeight_, qweight_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(float) * qbias_.count;
        CHECK(cudaMalloc((void **)&pGPUQBias_, size));
        CHECK(cudaMemcpy(pGPUQBias_, qbias_.values, size, cudaMemcpyHostToDevice));

        size = sizeof(half) * qweight16_.count;
        CHECK(cudaMalloc((void **)&pGPUQWeight16_, size));
        CHECK(cudaMemcpy(pGPUQWeight16_, qweight16_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(half) * qbias_.count;
        CHECK(cudaMalloc((void **)&pGPUQBias16_, size));
        CHECK(cudaMemcpy(pGPUQBias16_, qbias16_.values, size, cudaMemcpyHostToDevice));
    }

    {
        size = sizeof(float) * kweight_.count;
        CHECK(cudaMalloc((void **)&pGPUKWeight_, size));
        CHECK(cudaMemcpy(pGPUKWeight_, kweight_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(float) * kbias_.count;
        CHECK(cudaMalloc((void **)&pGPUKBias_, size));
        CHECK(cudaMemcpy(pGPUKBias_, kbias_.values, size, cudaMemcpyHostToDevice));

        size = sizeof(half) * kweight16_.count;
        CHECK(cudaMalloc((void **)&pGPUKWeight16_, size));
        CHECK(cudaMemcpy(pGPUKWeight16_, kweight16_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(half) * kbias16_.count;
        CHECK(cudaMalloc((void **)&pGPUKBias16_, size));
        CHECK(cudaMemcpy(pGPUKBias16_, kbias16_.values, size, cudaMemcpyHostToDevice));
    }

    {
        size = sizeof(float) * vweight_.count;
        CHECK(cudaMalloc((void **)&pGPUVWeight_, size));
        CHECK(cudaMemcpy(pGPUVWeight_, vweight_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(float) * vbias_.count;
        CHECK(cudaMalloc((void **)&pGPUVBias_, size));
        CHECK(cudaMemcpy(pGPUVBias_, vbias_.values, size, cudaMemcpyHostToDevice));

        size = sizeof(half) * vweight16_.count;
        CHECK(cudaMalloc((void **)&pGPUVWeight16_, size));
        CHECK(cudaMemcpy(pGPUVWeight16_, vweight16_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(half) * vbias16_.count;
        CHECK(cudaMalloc((void **)&pGPUVBias16_, size));
        CHECK(cudaMemcpy(pGPUVBias16_, vbias16_.values, size, cudaMemcpyHostToDevice));
    }

    {
        size = sizeof(float) * oweight_.count;
        CHECK(cudaMalloc((void **)&pGPUOWeight_, size));
        CHECK(cudaMemcpy(pGPUOWeight_, oweight_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(float) * obias_.count;
        CHECK(cudaMalloc((void **)&pGPUOBias_, size));
        CHECK(cudaMemcpy(pGPUOBias_, obias_.values, size, cudaMemcpyHostToDevice));

        size = sizeof(half) * oweight16_.count;
        CHECK(cudaMalloc((void **)&pGPUOWeight16_, size));
        CHECK(cudaMemcpy(pGPUOWeight16_, oweight16_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(half) * obias16_.count;
        CHECK(cudaMalloc((void **)&pGPUOBias16_, size));
        CHECK(cudaMemcpy(pGPUOBias16_, obias16_.values, size, cudaMemcpyHostToDevice));
    }

    {
        size = sizeof(float) * qkbias_.count;
        CHECK(cudaMalloc((void **)&pGPUQKBias_, size));
        CHECK(cudaMemcpy(pGPUQKBias_, qkbias_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(float) * qcbias_.count;
        CHECK(cudaMalloc((void **)&pGPUQCBias_, size));
        CHECK(cudaMemcpy(pGPUQCBias_, qcbias_.values, size, cudaMemcpyHostToDevice));

        size = sizeof(half) * qkbias16_.count;
        CHECK(cudaMalloc((void **)&pGPUQKBias16_, size));
        CHECK(cudaMemcpy(pGPUQKBias16_, qkbias16_.values, size, cudaMemcpyHostToDevice));
        size = sizeof(half) * qcbias16_.count;
        CHECK(cudaMalloc((void **)&pGPUQCBias16_, size));
        CHECK(cudaMemcpy(pGPUQCBias16_, qcbias16_.values, size, cudaMemcpyHostToDevice));
    }

    return 0;
}

void CSMHAPlugin::terminate() noexcept
{
    {
        CHECK(cudaFree(pGPUQWeight_));
        CHECK(cudaFree(pGPUQBias_));

        CHECK(cudaFree(pGPUQWeight16_));
        CHECK(cudaFree(pGPUQBias16_));
    }

    {
        CHECK(cudaFree(pGPUKWeight_));
        CHECK(cudaFree(pGPUKBias_));

        CHECK(cudaFree(pGPUKWeight16_));
        CHECK(cudaFree(pGPUKBias16_));
    }

    {
        CHECK(cudaFree(pGPUVWeight_));
        CHECK(cudaFree(pGPUVBias_));

        CHECK(cudaFree(pGPUVWeight16_));
        CHECK(cudaFree(pGPUVBias16_));
    }

    {
        CHECK(cudaFree(pGPUOWeight_));
        CHECK(cudaFree(pGPUOBias_));

        CHECK(cudaFree(pGPUOWeight16_));
        CHECK(cudaFree(pGPUOBias16_));
    }

    {
        CHECK(cudaFree(pGPUQKBias_));
        CHECK(cudaFree(pGPUQCBias_));

        CHECK(cudaFree(pGPUQKBias16_));
        CHECK(cudaFree(pGPUQCBias16_));
    }

    WHERE_AM_I()
}

void CSMHAPlugin::destroy() noexcept
{
    WHERE_AM_I();

    {
        free(const_cast<void *>(qweight_.values));
        free(const_cast<void *>(qbias_.values));

        free(const_cast<void *>(qweight16_.values));
        free(const_cast<void *>(qbias16_.values));
    }

    {
        free(const_cast<void *>(kweight_.values));
        free(const_cast<void *>(kbias_.values));

        free(const_cast<void *>(kweight16_.values));
        free(const_cast<void *>(kbias16_.values));
    }

    {
        free(const_cast<void *>(vweight_.values));
        free(const_cast<void *>(vbias_.values));

        free(const_cast<void *>(vweight16_.values));
        free(const_cast<void *>(vbias16_.values));
    }

    {
        free(const_cast<void *>(oweight_.values));
        free(const_cast<void *>(obias_.values));

        free(const_cast<void *>(oweight16_.values));
        free(const_cast<void *>(obias16_.values));
    }

    {
        free(const_cast<void *>(qkbias_.values));
        free(const_cast<void *>(qcbias_.values));

        free(const_cast<void *>(qkbias16_.values));
        free(const_cast<void *>(qcbias16_.values));
    }

    CHECK(cublasDestroy(handle_));
}

size_t CSMHAPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I()
    return sizeof(nK_) + sizeof(nN_)
        + sizeof(float) * qweight_.count + sizeof(float) * qbias_.count
        + sizeof(float) * kweight_.count + sizeof(float) * kbias_.count
        + sizeof(float) * vweight_.count + sizeof(float) * vbias_.count
        + sizeof(float) * oweight_.count + sizeof(float) * obias_.count
        + sizeof(float) * qkbias_.count + sizeof(float) * qcbias_.count;
}

void CSMHAPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I()
    char * data   = reinterpret_cast<char *>(buffer);
    size_t offset = 0;
    memcpy(data + offset, &nK_, sizeof(nK_));
    offset += sizeof(nK_);
    memcpy(data + offset, &nN_, sizeof(nN_));
    offset += sizeof(nN_);

    size_t size;

    {
        size = sizeof(float) * nK_ * nN_;
        memcpy(data + offset, qweight_.values, size);
        offset += size;
        size = sizeof(float) * nN_;
        memcpy(data + offset, qbias_.values, size);
        offset += size;
    }

    {
        size = sizeof(float) * nK_ * nN_;
        memcpy(data + offset, kweight_.values, size);
        offset += size;
        size = sizeof(float) * nN_;
        memcpy(data + offset, kbias_.values, size);
        offset += size;
    }

    {
        size = sizeof(float) * nK_ * nN_;
        memcpy(data + offset, vweight_.values, size);
        offset += size;
        size = sizeof(float) * nN_;
        memcpy(data + offset, vbias_.values, size);
        offset += size;
    }

    {
        size = sizeof(float) * nK_ * nN_;
        memcpy(data + offset, oweight_.values, size);
        offset += size;
        size = sizeof(float) * nN_;
        memcpy(data + offset, obias_.values, size);
        offset += size;
    }

    {
        size = sizeof(float) * nN_;
        memcpy(data + offset, qkbias_.values, size);
        offset += size;
        size = sizeof(float) * nN_;
        memcpy(data + offset, qcbias_.values, size);
        offset += size;
    }

}

void CSMHAPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}
const char *CSMHAPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *CSMHAPlugin::getPluginType() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_NAME;
}

const char *CSMHAPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

void CSMHAPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I()
    //handle_ = contextCublas;
}

void CSMHAPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

// class CSMHAPluginCreator

// Static class fields initialization
PluginFieldCollection    CSMHAPluginCreator::fc_ {};
std::vector<PluginField> CSMHAPluginCreator::attr_;

CSMHAPluginCreator::CSMHAPluginCreator()
{
    WHERE_AM_I()
    attr_.emplace_back(PluginField("k", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("n", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("qw", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("qb", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("kw", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("kb", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("vw", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("vb", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("ow", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("ob", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("qkb", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("qcb", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

CSMHAPluginCreator::~CSMHAPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2 *CSMHAPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I()

    printf("[CSMHA] [CSMHAPluginCreator] [createPlugin] %d\n",fc->nbFields);

    int     k, n;
    Weights qw, qb, kw, kb, vw, vb, ow, ob, qkb, qcb;
    for (int i = 0; i < fc->nbFields; i++)
    {
        PluginField field = fc->fields[i];
        std::string field_name(field.name);

        if (field_name.compare("qw") == 0)
        {
            qw.values = field.data;
            qw.count  = field.length;
            qw.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("qb") == 0)
        {
            qb.values = field.data;
            qb.count  = field.length;
            qb.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("kw") == 0)
        {
            kw.values = field.data;
            kw.count  = field.length;
            kw.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("kb") == 0)
        {
            kb.values = field.data;
            kb.count  = field.length;
            kb.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("vw") == 0)
        {
            vw.values = field.data;
            vw.count  = field.length;
            vw.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("vb") == 0)
        {
            vb.values = field.data;
            vb.count  = field.length;
            vb.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("ow") == 0)
        {
            ow.values = field.data;
            ow.count  = field.length;
            ow.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("ob") == 0)
        {
            ob.values = field.data;
            ob.count  = field.length;
            ob.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("qkb") == 0)
        {
            qkb.values = field.data;
            qkb.count  = field.length;
            qkb.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("qcb") == 0)
        {
            qcb.values = field.data;
            qcb.count  = field.length;
            qcb.type   = DataType::kFLOAT;
            continue;
        }
        if (field_name.compare("k") == 0)
        {
            k = *reinterpret_cast<const int *>(field.data);
        }
        if (field_name.compare("n") == 0)
        {
            n = *reinterpret_cast<const int *>(field.data);
        }
    }
    return new CSMHAPlugin(name, qw, qb, kw, kb, vw, vb, ow, ob, qkb, qcb, k, n);
}

IPluginV2 *CSMHAPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I()
    return new CSMHAPlugin(name, serialData, serialLength);
}

void CSMHAPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}

const char *CSMHAPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *CSMHAPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_NAME;
}
const char *CSMHAPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

const PluginFieldCollection *CSMHAPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I()
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(CSMHAPluginCreator);

} // namespace nvinfer1
