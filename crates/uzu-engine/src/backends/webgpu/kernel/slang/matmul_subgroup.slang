// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, a_group_size)
UZU_PARAM(bool, a_grouped)
UZU_PARAM(bool, a_int8)
UZU_PARAM(bool, b_quantized)
UZU_PARAM(uint, b_stride)
UZU_PARAM(uint, bits)
UZU_PARAM(uint, correction)
UZU_PARAM(bool, gather)
UZU_PARAM(uint, group_size)
UZU_PARAM(bool, has_bias)
UZU_PARAM(bool, has_soft_cap)
UZU_PARAM(uint, k)
UZU_PARAM(uint, n)
UZU_PARAM(float, output_scale)
UZU_PARAM(uint, row_stride)
UZU_PARAM(uint, scale_group_stride)
UZU_PARAM(uint, scale_output_stride)
UZU_PARAM(bool, signed_codes)
UZU_PARAM(float, soft_cap)
UZU_PARAM(bool, transpose)
UZU_PARAM(uint, zp_group_stride)
UZU_PARAM(uint, zp_output_stride)
UZU_BUFFER(a)
UZU_BUFFER(a_scales)
UZU_BUFFER(b)
UZU_BUFFER(bias)
UZU_BUFFER(codes)
UZU_BUFFER(d)
UZU_BUFFER(gather)
UZU_BUFFER(quant_biases)
UZU_BUFFER(scales)
UZU_BUFFER(zero_points)

func read_a(row:uint,inner:uint)->float {
 if(!param_a_int8()){return load_a(row*param_k()+inner);}
 var code=inner;if(param_a_grouped()){let offset=inner%8u;code=inner-offset+(offset%2u)*4u+offset/2u;}
 return load_a(row*param_k()+code)*load_a_scales(row*(param_k()/param_a_group_size())+inner/param_a_group_size());
}
func quant_params(column:uint,group:uint)->float2 {
 let metadata=column*param_scale_output_stride()+group*param_scale_group_stride();let scale=load_scales(metadata);var correction=-scale*float(1u<<(param_bits()-1u));
 if(param_correction()==1u){correction=load_quant_biases(metadata);}
 if(param_correction()==2u){let pointIndex=column*param_zp_output_stride()+group*param_zp_group_stride();let point=(word_zero_points(pointIndex/(32u/param_bits()))>>((pointIndex%(32u/param_bits()))*param_bits()))&((1u<<param_bits())-1u);correction=-scale*float(point);}
 return float2(scale,correction);
}
func read_b(column:uint,inner:uint)->float {
 if(!param_b_quantized()){if(param_transpose()){return load_b(column*param_b_stride()+inner);}return load_b(inner*param_b_stride()+column);}
 let packed=32u/param_bits();let index=column*param_k()+inner;var code=(word_codes(index/packed)>>((index%packed)*param_bits()))&((1u<<param_bits())-1u);
 if(param_signed_codes()){code^=1u<<(param_bits()-1u);}let quant=quant_params(column,inner/param_group_size());return quant.x*float(code)+quant.y;
}
func write_output(row:uint,column:uint,accumulator:float) {
 var value=accumulator*param_output_scale();if(param_has_bias()){value+=load_bias(column);}if(param_has_soft_cap()){value=param_soft_cap()*tanh(value/param_soft_cap());}store_d(row*param_row_stride()+column,value);
}

[shader("compute")]
[numthreads(128,1,1)]
void main(uint3 group: SV_GroupID, uint lane: SV_WaveLaneIndex, uint width: SV_WaveLaneCount, uint subgroup: SV_WaveIndex) {
let subgroups = (128u + width - 1u) / width;
    let row=group.y;
    let baseColumn=(group.x+group.z*65535u)*4u;
    for(var slot=0u;slot<4u;slot+=subgroups){
        let column=baseColumn+slot+subgroup;
        let valid=slot+subgroup<4u && column<param_n();
        var sum=0.0;
        if(valid){
            var weightColumn=column;
            if(param_gather()){weightColumn=word_gather(row*param_n()+column);}
            let packed=32u/param_bits();
            for(var word=lane;word<param_k()/packed;word+=width){
                let inner=word*packed;
                let codes=word_codes(weightColumn*(param_k()/packed)+word);
                let quant=quant_params(weightColumn,inner/param_group_size());

var code0=(codes>>(0u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code0^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+0u)*(quant.x*float(code0)+quant.y);
var code1=(codes>>(1u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code1^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+1u)*(quant.x*float(code1)+quant.y);
var code2=(codes>>(2u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code2^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+2u)*(quant.x*float(code2)+quant.y);
var code3=(codes>>(3u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code3^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+3u)*(quant.x*float(code3)+quant.y);
if(param_bits()==4u){
var code4=(codes>>(4u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code4^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+4u)*(quant.x*float(code4)+quant.y);
var code5=(codes>>(5u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code5^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+5u)*(quant.x*float(code5)+quant.y);
var code6=(codes>>(6u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code6^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+6u)*(quant.x*float(code6)+quant.y);
var code7=(codes>>(7u*param_bits()))&((1u<<param_bits())-1u);
if(param_signed_codes()){code7^=1u<<(param_bits()-1u);}
sum+=read_a(row,inner+7u)*(quant.x*float(code7)+quant.y);
}
            }
        }
        let total=WaveActiveSum(sum);
        if(valid && lane==0u){write_output(row,column,total);}
    }
}
