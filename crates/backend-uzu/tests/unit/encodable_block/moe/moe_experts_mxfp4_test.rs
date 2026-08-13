use half::bf16;
use proc_macros::uzu_test;

use super::{MoeExpertsMxfp4Arguments, MoeExpertsMxfp4DecodeBlock, MoeExpertsMxfp4PrefillBlock};
use crate::{
    backends::common::Encoder,
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_prefix_to_vec, create_context, for_each_non_cpu_backend},
    },
};

const D_MODEL: usize = 32;
const D_FF: usize = 4;

// Captured from openai/gpt-oss-20b revision 6cee5e81ee83917806bbde320786a8fb61efebee,
// model-00000-of-00002.safetensors (SHA-256 16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427).
// The fixture takes expert 0, the first input group, and the smallest row slices
// that form a complete packed two-projection expert. INPUT is a real BF16 slice
// from model.layers.0.input_layernorm.weight used as a stable probe vector.
const INPUT_BF16: [u8; 64] = [
    0xb2, 0x3f, 0xa1, 0x3f, 0x8a, 0x3f, 0x6e, 0x40, 0x94, 0x3f, 0x2e, 0x3f, 0x00, 0x40, 0x03, 0x40, 0x51, 0x3f, 0x83,
    0x3f, 0x8b, 0x40, 0xba, 0x3f, 0x7b, 0x3f, 0x35, 0x3f, 0x70, 0x40, 0x18, 0x40, 0x44, 0x40, 0x98, 0x40, 0x96, 0x3f,
    0xc4, 0x40, 0xa2, 0x3f, 0xa7, 0x40, 0x7e, 0x3f, 0xa3, 0x40, 0x94, 0x40, 0x4d, 0x40, 0x2c, 0x40, 0x08, 0x3f, 0x13,
    0x40, 0xb6, 0x3f, 0x7c, 0x3f, 0xbb, 0x3f,
];

// Checkpoint order is [gate0, up0, gate1, up1, ...].
const GATE_UP_BLOCKS_CHECKPOINT: [u8; 128] = [
    0x00, 0xc0, 0x80, 0xa9, 0x10, 0x1b, 0x81, 0x22, 0x93, 0xe4, 0xa0, 0xb2, 0x3b, 0x19, 0xb2, 0x31, 0x82, 0xf8, 0xce,
    0xa8, 0x6e, 0x82, 0xa2, 0x0f, 0xc5, 0x76, 0xc2, 0x7a, 0xcc, 0x8c, 0xc2, 0x24, 0x45, 0x65, 0xdd, 0xf5, 0xc2, 0x29,
    0x81, 0xf5, 0x66, 0xe7, 0x43, 0xca, 0x04, 0xb1, 0x52, 0x91, 0x04, 0x0a, 0x58, 0x44, 0x68, 0x45, 0x82, 0xc7, 0x62,
    0x26, 0x7a, 0x78, 0x8d, 0x02, 0x40, 0x0c, 0x4e, 0xc6, 0xa8, 0x5d, 0xa2, 0x45, 0x0a, 0x26, 0xaa, 0xd2, 0x44, 0xe7,
    0x56, 0x47, 0x2e, 0xdf, 0x5b, 0xcc, 0xa9, 0xad, 0x59, 0xe4, 0x03, 0x51, 0x4d, 0xe4, 0x54, 0x58, 0x29, 0x1a, 0x28,
    0xaa, 0x11, 0x21, 0x18, 0x12, 0xcc, 0x25, 0x00, 0xc8, 0x08, 0xe1, 0x88, 0x3b, 0x52, 0xa1, 0x0c, 0x13, 0xc5, 0x6d,
    0xd6, 0x6a, 0xde, 0x6d, 0x28, 0x0a, 0xcc, 0xc8, 0xfa, 0xdd, 0x6d, 0xa2, 0xf6, 0xac,
];
const GATE_UP_SCALES_CHECKPOINT: [u8; 8] = [0x7a, 0x79, 0x78, 0x79, 0x78, 0x79, 0x79, 0x79];
const GATE_UP_BIASES_CHECKPOINT: [u8; 16] =
    [0x3e, 0xbf, 0x46, 0xbf, 0xd6, 0xbe, 0x67, 0xbf, 0xfa, 0xbe, 0x65, 0xbf, 0xc7, 0xbe, 0x50, 0xbf];

const DOWN_BLOCKS: [u8; 512] = [
    0x7b, 0xdc, 0x22, 0x5a, 0x31, 0xba, 0xc2, 0x18, 0x01, 0x4a, 0xea, 0x22, 0x44, 0xe9, 0x21, 0x82, 0x10, 0x9c, 0x41,
    0x69, 0x3d, 0x44, 0x59, 0x1d, 0xcf, 0x52, 0xd9, 0x99, 0x63, 0xec, 0x2b, 0xd8, 0xac, 0x28, 0xf9, 0x2d, 0x05, 0xaa,
    0x13, 0xc4, 0xa1, 0xde, 0x9d, 0xb2, 0x10, 0xc1, 0x4e, 0xd2, 0xea, 0x84, 0xe0, 0x43, 0x2c, 0x91, 0x1c, 0x31, 0x3d,
    0x2b, 0xc9, 0x22, 0x9a, 0xb5, 0x3a, 0x14, 0xae, 0x86, 0xe7, 0xac, 0xe7, 0xa6, 0x5d, 0x2d, 0xc7, 0x87, 0xa6, 0x24,
    0xaa, 0xa2, 0xcd, 0x8c, 0x7d, 0xa0, 0x66, 0xc5, 0x8a, 0x26, 0xef, 0x84, 0x4c, 0x24, 0xc4, 0xc0, 0x2a, 0x6d, 0x22,
    0x2e, 0x1a, 0xc2, 0xa0, 0x12, 0xbd, 0x5b, 0xa9, 0x2a, 0x3c, 0x38, 0x9d, 0x41, 0x5a, 0xa0, 0xd6, 0xa2, 0x04, 0xa1,
    0x53, 0x65, 0x03, 0x9a, 0x56, 0x65, 0xc4, 0x47, 0x8b, 0x29, 0x53, 0xd9, 0xa4, 0xed, 0x39, 0xec, 0x1c, 0x89, 0xa8,
    0xd3, 0x30, 0xa4, 0xaf, 0x30, 0x89, 0xca, 0xb3, 0xbe, 0x51, 0x2c, 0x26, 0xad, 0xb4, 0xc3, 0x96, 0x9e, 0xc1, 0xbd,
    0xbf, 0xd4, 0x5a, 0xb9, 0xdb, 0x1e, 0x50, 0x2d, 0x13, 0x24, 0x4a, 0xa4, 0x32, 0xaf, 0xcd, 0x1e, 0x47, 0x4a, 0x8c,
    0x5a, 0xab, 0x56, 0x45, 0x80, 0x84, 0xc5, 0x67, 0x74, 0x2a, 0xd2, 0xe4, 0xca, 0xe0, 0x20, 0xa4, 0x28, 0x50, 0x57,
    0x6e, 0xe0, 0xac, 0xac, 0xa2, 0xdc, 0x84, 0x04, 0x04, 0x7d, 0xad, 0xd6, 0xa5, 0x5e, 0xa2, 0x0c, 0xe8, 0x5a, 0x11,
    0x84, 0x9d, 0x4b, 0x89, 0x91, 0xe9, 0xa9, 0x8c, 0x81, 0x3b, 0xb0, 0x08, 0x1b, 0x28, 0x9a, 0x09, 0xd5, 0x6a, 0xb1,
    0x44, 0x92, 0xcc, 0x54, 0x84, 0xda, 0x59, 0x34, 0x1b, 0xbc, 0x19, 0xa1, 0xdc, 0x8a, 0x84, 0xd9, 0x48, 0x43, 0xbd,
    0x48, 0xb5, 0x54, 0x04, 0x16, 0x46, 0xca, 0xaa, 0x63, 0x5a, 0xbb, 0x34, 0x12, 0x0d, 0x12, 0xf5, 0x9a, 0x09, 0x99,
    0x53, 0x99, 0x3b, 0x3e, 0x92, 0x61, 0x18, 0xc0, 0x08, 0xa9, 0x9c, 0x12, 0x21, 0xc6, 0xba, 0x81, 0x12, 0x99, 0xb9,
    0x24, 0x28, 0x1b, 0xba, 0xd5, 0x99, 0x6c, 0xc2, 0xe8, 0xd8, 0x47, 0xb5, 0x12, 0xeb, 0x9a, 0x94, 0x49, 0x4c, 0xcd,
    0x1d, 0xfe, 0xd2, 0x43, 0x5a, 0x2e, 0xe9, 0x9e, 0xec, 0x65, 0x90, 0x46, 0x51, 0xb5, 0x95, 0xec, 0x06, 0x54, 0x28,
    0x7c, 0x5a, 0xdf, 0xf5, 0x74, 0x88, 0x8e, 0xaa, 0xc8, 0x4e, 0xdc, 0x84, 0x7a, 0x1c, 0xb9, 0x66, 0x2c, 0x1e, 0x98,
    0x0a, 0xa4, 0x69, 0x6a, 0x34, 0xa6, 0xd3, 0x59, 0x44, 0x9d, 0xe7, 0x66, 0x5e, 0xa8, 0x62, 0x42, 0x62, 0x07, 0x65,
    0xed, 0xe4, 0x28, 0xdc, 0x2f, 0xfd, 0xe7, 0xcb, 0x5e, 0x3a, 0xe4, 0x35, 0x9d, 0x54, 0xa2, 0xbd, 0xa4, 0xeb, 0x99,
    0xeb, 0x50, 0xce, 0x2b, 0x05, 0x41, 0x2c, 0x44, 0x5c, 0x19, 0xdd, 0x61, 0x56, 0x38, 0x2c, 0x65, 0x62, 0x37, 0x32,
    0x29, 0xc9, 0x94, 0x3d, 0x0b, 0x19, 0x5b, 0xe0, 0x33, 0xcd, 0x42, 0x1d, 0xa9, 0x16, 0x29, 0x6d, 0x39, 0xfa, 0xdf,
    0x6d, 0xd6, 0x75, 0xce, 0x4e, 0x15, 0x23, 0xc6, 0x08, 0xa3, 0x4a, 0x63, 0xe6, 0x6a, 0xd8, 0xa4, 0x1c, 0x94, 0x4b,
    0x0a, 0xb3, 0x95, 0xdb, 0xde, 0x1a, 0xae, 0xa9, 0xa3, 0x99, 0x49, 0x60, 0x3b, 0x12, 0x4c, 0x51, 0x94, 0x30, 0x0a,
    0x6a, 0x3d, 0x55, 0xb9, 0x31, 0x14, 0x25, 0x49, 0x7d, 0x08, 0x8a, 0xae, 0xd7, 0xdf, 0xa2, 0x67, 0x52, 0xc6, 0x54,
    0x04, 0xea, 0x84, 0xc5, 0xae, 0x4e, 0xac, 0x74, 0xb0, 0xb8, 0x44, 0x2a, 0xc1, 0x16, 0xb9, 0x30, 0x34, 0xed, 0x91,
    0x6c, 0x42, 0xc6, 0xf0, 0xd2, 0x40, 0x44, 0x4a, 0xe8, 0xae, 0xa6, 0x5a, 0x24, 0xe7, 0xe7, 0x87, 0x0f, 0x82,
];
const DOWN_SCALES: [u8; 32] = [
    0x79, 0x79, 0x79, 0x78, 0x79, 0x7b, 0x78, 0x78, 0x7b, 0x7a, 0x77, 0x78, 0x7a, 0x7c, 0x78, 0x79, 0x78, 0x78, 0x79,
    0x76, 0x7a, 0x77, 0x7a, 0x77, 0x77, 0x78, 0x77, 0x7c, 0x78, 0x79, 0x79, 0x79,
];
const DOWN_BIASES_BF16: [u8; 64] = [
    0x05, 0x3d, 0x74, 0xbc, 0x0f, 0xbd, 0x52, 0xbc, 0xed, 0xbd, 0x11, 0x3e, 0x18, 0x3b, 0xe4, 0xbc, 0x8f, 0x3e, 0xa7,
    0xbd, 0x04, 0xbd, 0x0d, 0x3d, 0xb5, 0x3e, 0xae, 0x3e, 0x91, 0x3c, 0x39, 0xbc, 0xa5, 0xbc, 0xed, 0x3c, 0x3f, 0xbd,
    0x2d, 0x3b, 0xed, 0x3e, 0xa4, 0xbc, 0xd8, 0x3d, 0xf7, 0xbb, 0xa8, 0xbc, 0xb4, 0x3c, 0x9b, 0x3b, 0x9b, 0x3c, 0x12,
    0xbc, 0xc4, 0xbd, 0x84, 0x3d, 0x8d, 0xbd,
];

// The multigroup fixture extends the same expert-0 capture to d_model=64 and
// d_ff=32: 64 canonical gate/up rows across two input groups, plus 64 down
// rows across one complete hidden group. Per-slice hashes make recapture drift
// visible without requiring the 4.8 GB source shard in the test environment.
// 128 bytes; SHA-256 fafbda6e284ed187210994f22df05e35eaff4296c4b4b0f2a9e6107b58ee4293
const MULTIGROUP_INPUT_BF16_HEX: &str = concat!(
    "b23fa13f8a3f6e40943f2e3f00400340513f833f8b40ba3f7b3f353f7040184044409840963fc440a23fa7407e3fa34094404d402c40083f1340b63f7c3fbb3f",
    "614008406b40793f3840114049406a40a4403040d83f104062408a404140243f8340e73f013f05408c3f3140f83f0a40683f7e3f02409140c43f7140383fa43f",
);

// 2048 bytes; SHA-256 7a32089353adea970817b49379aef4b9474da270dfe0040c56e80362dbc60a22
const MULTIGROUP_GATE_UP_BLOCKS_HEX: &str = concat!(
    "00c080a9101b812293e4a0b23b19b231867222d2654a2288667a4dc2d2ce654a82f8cea86e82a20fc576c27acc8cc224a256407ecaea24dfcae28576ce0d7dca",
    "4565ddf5c22981f566e743ca04b152912cacd9a2d990c1113971cd010ccb5240040a5844684582c762267a788d02400c2dbe232c66ba4dad2c109c0f1b3d59f8",
    "4ec6a85da2450a26aad244e756472edf2b0c11252ec9312c0b48914c82345b085bcca9ad59e403514de45458291a28aa1928bc891115a89aeaa0118598d3b101",
    "11211812cc2500c808e1883b52a10c1311a2918db4a8ea05aa28491c8192aae0c56dd66ade6d280accc8fadd6da2f6aca59ed4213adaed818649314c8b246e28",
    "8250f8ad82820a06f4a0c202ae0acf87a345ccabeb23a88ab530b34589c41a90182834b3b84081b54b632bda8c918c32852d3a3cac93018ae6911214ac5194a8",
    "98da11a000b9884f242bd45dd42434aac8cf2fad86c000c7a2d8f66770504ce054ea99aa99138048aa142a49918b9aaa9b0ca9931289932bc1909ba64c99cd98",
    "1d3e13b6539e810352abc159b3a509b8c251c3b945da00ab1a30e2359acd3bd818081b93111b00bba468041a0108a383a4ad28fa8e2a2a8aae087e82058aefc8",
    "822c1b946c120129585a9a2cc4a8ab08be2308324fdf2124143aa9a311434cc0c9b11915d3b90963c4a1756a5e8924c4911690562caa318fa1b8bc2519892390",
    "a9b2b112451d01c06aa3a1816922a2919525ce56891b22254c182a50a3abb4c0b119901193808152928544109615208b333031c1829a305bcc11028c293645a8",
    "a4a33364498c011130bd99e94b8d4db49da298b1b14126462e98d2b2b92288e9ae5008a5233c01cc4cd9a08130944a24dbb219f4fec73702897120c3e24dfd28",
    "2b21b120b92e88e2db5a9c0cb58d39c1bb323c4994c214162bb08101425981b80adaa0ffd22a0ac6cccd588224a2d50eaf6a4d5e0227a5d62fa80c0ed2d4c740",
    "d1b9294c1b03009cd2354ae12918aca9999a990230889d09479139121a2c61485b19ab536f3511e6199fc9a12eae5902aa8b9419d34a5c0ea981aa8b904519b0",
    "582121b2104988a4cb38b2d3ac86999619c6e01a5d3c32293180a929598d52990c3ac272ba6c8a35404054682fb45596213cc53fc5dd4b5838c91b5f4c5d4120",
    "04da442242ea02cd4870285ae70a8a8ab3b4119b15db5419ead1bbd3db62b999dae0545c54831197abf5387922b0a92e0dd2fcd02082255048d04f22aa640520",
    "7c65cc4fd2c68af27e7c76ec74848e8a173bca31c4b15813e6a9c1b5da391999ec8c4c5a2a4c8826da76a8420a2f8c20102b893bc33a5dbab288a801896108a9",
    "90d91252a220099ec2d9ba599c9c2d61235325d3b51694c49049a984c28d0820433128292ac8090ba93aea3103181999a224dcc7cc457ec08d5a440220d026a0",
    "101418c5c9c402d6cc4139121d96bb9d8cb241b5bb80fd87aa311922de155d80a1c0a24da1d1902698689adb501b39cf829a6de5bec11c2914a213a91380ab90",
    "58a1925319ba094a0b83abd97d9c8d1155d9cb6ca4c2251c3cc9c298829451811c69b29a6bda980b8ee3909b9b9b62aa0a555fc2e4e7c42c40c0dad58cd77202",
    "e852a276eac8025e7677227a6c87cdafbc212910e7a1c4035a21b103bcb5cbe01ccc1b4519250104c1864ef0d616221e3a902e3a4f4a930e92c038a929009090",
    "285d436c92d28102b5522c1c93262198aa27246ee2a2d84226d842224a5c6c5820ac92bac19099a200b853e21a2182095a2aaf74a7d462aca2dafc45a6e4c240",
    "ca2ed2c7ea4d85ac76eaa055f4accf46ba2131919ac893b2162039094ac2b938da528a20c22f00ad78ea6a488080aa2244a8454f50e42dc7062850482026caaa",
    "2212911b992f8156af4391c93c934a148530dd64a0c861bcd9b9243b21e5e941890b11144a9b08b53043923c4e18a9a3c22c872d8d467ae4e46aef0d4226ecd0",
    "d10c2119341b9981095dfbd85a05c00c828c9696cb093c93dca1311ac9d36e102174c062d6d91a4529995b9124119aa2a221a9935422b5092529a2b69982c220",
    "025a880119d888c98ae209acbb01b41100cc2c88c4cccb0e9830e13232a25990ada49a1b1bab898899e6c4809912ba1015da15552c333ca329c9f2bc5ca55d4a",
    "4b94a2f1494110b1cdd868c311aa90cc99135d31e0dc02a5044842a19b622e58ccea2588ceaa022c57cef842cf2d4a5c9545d10ba620bc8c35c1c91398452b20",
    "3212382c41b5918902233ac98e05ad2c8dab5ccba33c64183211a5530a115149c4a430d0504f0956ed32bcdebc160f95d5240272ef02a0245aaac05a0ad6a452",
    "a8231a1a331481806de35c29cd0b1cb993a6eb2caeb2430dc332189d9440c8c931e01c33b51592cd0ed5e199d4091187525c09c4c295ca30763a5d66a2445e2a",
    "0f4d4842dac284864e68a2e02725ce472d00a84002913a1c2399c22282a6b320428b14ce9b5c102b945f986b500bb4919acfc284863aa5a106e2c8ea42112cd1",
    "42614451d22e11ac5f3cb3423fbabaa1b4ebe55a498bad95dcd169e7029c35b0ca9a91552292880422451319cf0944020c1d84ccc9411701a949d1c93aa2da49",
    "d44cacd6a4d705e50cf4445a2cad6d229a10a1ab9dba5132b4a2624264cc2218862e12c84946930c8d9449e46905b32cb41b121aa24a223b44604a31112b1928",
    "64d130294a20892d91431349cb81981531baa94d1d802eaab358c5a49141bc50681e95421c6e90a6aa81dc81d1878c9e19338b21be109015931a1a2cba915110",
    "ac0acd24082882aa8e722244e226a8280a93aa85b5cb40265411598d9a4ed919f6134946302c0b65e6de65c4b614bbc11a093243194954c289a0394bdb826cb8",
);

// 128 bytes; SHA-256 355f9ba021dba78991e6192e5c29c9054a0b19ad20a476811a687814f350584b
const MULTIGROUP_GATE_UP_SCALES_HEX: &str = concat!(
    "7a787979787979797879797a7979797a78797a7a79787a7a79797a797979797a79797a7a79797a79797979797979797a79797979797979797879797a79797a79",
    "7979797a797979797879797a79797a797879797979797a797979797a79797a797979797a79797979797979797879797979797a7a7879797a7979797a7979797a",
);

// 128 bytes; SHA-256 7d20460fac732a4491d43f3593719df16df30ca8044a718548d5822ee2dd1142
const MULTIGROUP_GATE_UP_BIASES_BF16_HEX: &str = concat!(
    "3ebf46bfd6be67bffabe65bfc7be50bfbcbe69bf13bf2cbfb6be82bffabe63bf10bf49bf17bf76bfe9be45bf25bf52bfcebe75bf0abf50bf03bf82bfa0be46bf",
    "10bf7abfd9be74bf04bf4cbf47bf59bf94be38bfc5be66bfe6be7ebfd5be6dbfa0be78bfffbe78bfbcbe69bfc1bd82bf1ebf77bfe0be48bfe8bc4ebfb0be54bf",
);

// 1024 bytes; SHA-256 0b2527d6952156a2bc676e4a270aebc38f8d706ab5f62e42dbd5ed3bec10b006
const MULTIGROUP_DOWN_BLOCKS_HEX: &str = concat!(
    "7bdc225a31bac218014aea2244e92182109c41693d44591dcf52d99963ec2bd8ac28f92d05aa13c4a1de9db210c14ed2ea84e0432c911c313d2bc9229ab53a14",
    "ae86e7ace7a65d2dc787a624aaa2cd8c7da066c58a26ef844c24c4c02a6d222e1ac2a012bd5ba92a3c389d415aa0d6a204a15365039a5665c4478b2953d9a4ed",
    "39ec1c89a8d330a4af3089cab3be512c26adb4c3969ec1bdbfd45ab9db1e502d13244aa432afcd1e474a8c5aab56458084c567742ad2e4cae020a42850576ee0",
    "acaca2dc8404047dadd6a55ea20ce85a11849d4b8991e9a98c813bb0081b289a09d56ab14492cc5484da59341bbc19a1dc8a84d94843bd48b554041646caaa63",
    "5abb34120d12f59a099953993b3e926118c008a99c1221c6ba811299b924281bbad5996cc2e8d847b512eb9a94494ccd1dfed2435a2ee99eec65904651b595ec",
    "0654287c5adff574888eaac84edc847a1cb9662c1e980aa4696a34a6d359449de7665ea86242620765ede428dc2ffde7cb5e3ae4359d54a2bda4eb99eb50ce2b",
    "05412c445c19dd6156382c6562373229c9943d0b195be033cd421da916296d39fadf6dd675ce4e1523c608a34a63e66ad8a41c944b0ab395dbde1aaea9a39949",
    "603b124c5194300a6a3d55b9311425497d088aaed7dfa26752c65404ea84c5ae4eac74b0b8442ac116b93034ed916c42c6f0d240444ae8aea65a24e7e7870f82",
    "b3ac12b01d6d2ba6b9901cbb80c05a1d2984a21a51c240846b521355c2244ba1d582422257688246a54248a24d2e22ca56d2eabc8be9df9c1a914dd83a2ad814",
    "8a11c4b430384a158cd33c859d18a246ccc5200aa422a22085e02d40cca6467aab01a2b2943a50b0270412403bc10a49d8660eac7a2a8ecccda24474c0ad2e5a",
    "1b0a4ca028a94da45641eba5c8c041b9c55032cca1a98b9a90bc89e39b1311c18c5a9b147dc4646ac95bced356b7ad99b4552c0c4ed5cd55e974da6303d8a016",
    "138e2a251155a3932029b3740b8199992270a6ad52a8da22e260ceac428842ccb52a1c615c302490ab5d2239bd39da9c189d53d899e02aa488290d9d2cba019d",
    "0444893b6a18b14c9adc5241912998ac36289b481a39119c9ac2a92abe89351392d3b1b4ab19b88e923398ab3a2951a428a3824a0c211892e2429204440a2981",
    "92619c23d5631ae1690c365dd96b1a513dea4ee248e1058ec41da6ded49c5622a15544255e6bbd5aac95cee4e2549d4a5a0afd2722428c72d66caadd02022080",
    "c5939c1ac13d693ca2135ca112941cb29bcedbdaaaab59c5c91493d33a22b911ac20ed84440ed02800222aaa48a207adc5ee890e62d27ac644e54713cc97a670",
    "a3e5ad5adc9d5d3a4c994e398b4e5c11964132219294401812cbbb23a29a2c9c93d41832899352528641ac21ac81dbbbac44055ca2a4a2ca8aa268728aa2d558",
);

// 64 bytes; SHA-256 53f8fcb3c56234db5f8c9649c51ee10b307dd1bcd4a649bb7099188a8e5c7218
const MULTIGROUP_DOWN_SCALES_HEX: &str = "79797978797b78787b7a77787a7c7879787879767a777a777778777c78797979787a777a78787877777878787877787b78797c79797778787a7a787778787b79";

// 128 bytes; SHA-256 c3e120bc74bf9251f35b00a216a9ebeb3d3ffda950fc1b82c32b03ee6646414e
const MULTIGROUP_DOWN_BIASES_BF16_HEX: &str = concat!(
    "053d74bc0fbd52bcedbd113e183be4bc8f3ea7bd04bd0d3db53eae3e913c39bca5bced3c3fbd2d3bed3ea4bcd83df7bba8bcb43c9b3b9b3c12bcc4bd843d8dbd",
    "51bcf03cf23ce53dbabcc0bd1bbd673cbfbbecbc403d623d47bb01bc083cbebd64bb84bcc33d72bd1d3ece3beabb98bc353ea23d8a3c0bbd573de4bc513e7a3c",
);

fn bf16_from_le_bytes(bytes: &[u8]) -> Vec<bf16> {
    let (bytes, remainder) = bytes.as_chunks::<2>();
    assert!(remainder.is_empty(), "BF16 fixture must contain complete values");
    bytes.iter().map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes))).collect()
}

fn bytes_from_hex(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "hex fixture must contain complete bytes");
    hex.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|digits| {
            let decode = |digit| match digit {
                b'0'..=b'9' => digit - b'0',
                b'a'..=b'f' => digit - b'a' + 10,
                _ => panic!("fixture contains a non-hex digit"),
            };
            decode(digits[0]) << 4 | decode(digits[1])
        })
        .collect()
}

/// Lalamo's canonical group-16 gate/up payload, derived losslessly from checkpoint pairs.
fn stack_gate_up_mxfp4(
    blocks: &[u8],
    scales: &[u8],
    biases: &[bf16],
    d_model: usize,
    d_ff: usize,
) -> (Vec<u8>, Vec<u8>, Vec<bf16>) {
    let bytes_per_row = d_model / 2;
    let checkpoint_groups_per_row = d_model / 32;
    let rows = (0..d_ff).map(|row| 2 * row + 1).chain((0..d_ff).map(|row| 2 * row)).collect::<Vec<_>>();
    let blocks =
        rows.iter().flat_map(|&row| blocks[row * bytes_per_row..(row + 1) * bytes_per_row].iter().copied()).collect();
    let scales = rows
        .iter()
        .flat_map(|&row| {
            scales[row * checkpoint_groups_per_row..(row + 1) * checkpoint_groups_per_row]
                .iter()
                .flat_map(|&scale| [scale, scale])
        })
        .collect();
    let biases = rows
        .iter()
        .enumerate()
        .map(|(row, &checkpoint_row)| {
            let bias = biases[checkpoint_row];
            if row < d_ff {
                return bf16::from_f32(f32::from(bias) + 1.0);
            }
            bias
        })
        .collect();
    (blocks, scales, biases)
}

fn decode_mxfp4(
    blocks: &[u8],
    scales: &[u8],
    row: usize,
    row_width: usize,
    column: usize,
) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

    let groups_per_row = row_width.div_ceil(32);
    let group = column / 32;
    let element = column % 32;
    let packed = blocks[(row * groups_per_row + group) * 16 + element / 2];
    let code = if element.is_multiple_of(2) {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let exponent = i32::from(scales[row * groups_per_row + group]) - 127;
    VALUES[usize::from(code)] * 2.0f32.powi(exponent)
}

fn dense_reference(
    input: &[bf16],
    w13_blocks: &[u8],
    w13_scales: &[u8],
    w2_blocks: &[u8],
    w2_scales: &[u8],
    w13_global_scale: f32,
    w2_global_scale: f32,
    up_biases: &[bf16],
    down_biases: &[bf16],
    d_model: usize,
    d_ff: usize,
) -> Vec<bf16> {
    let mut hidden = vec![0.0f32; d_ff];
    for hidden_idx in 0..d_ff {
        let gate_row = 2 * hidden_idx;
        let up_row = gate_row + 1;
        let mut gate = 0.0f32;
        let mut up = 0.0f32;
        for model_idx in 0..d_model {
            let input = f32::from(input[model_idx]);
            gate = input.mul_add(decode_mxfp4(w13_blocks, w13_scales, gate_row, d_model, model_idx), gate);
            up = input.mul_add(decode_mxfp4(w13_blocks, w13_scales, up_row, d_model, model_idx), up);
        }
        gate = gate * w13_global_scale + f32::from(up_biases[gate_row]);
        up = up * w13_global_scale + f32::from(up_biases[up_row]);
        let up = up.clamp(-7.0, 7.0) + 1.0;
        let gate = gate.min(7.0);
        hidden[hidden_idx] = (gate / (1.0 + (-1.702 * gate).exp())) * up;
    }

    (0..d_model)
        .map(|model_idx| {
            let mut output = 0.0f32;
            for hidden_idx in 0..d_ff {
                output =
                    hidden[hidden_idx].mul_add(decode_mxfp4(w2_blocks, w2_scales, model_idx, d_ff, hidden_idx), output);
            }
            output = output * w2_global_scale + f32::from(down_biases[model_idx]);
            bf16::from_f32(output)
        })
        .collect()
}

#[uzu_test]
fn test_real_gpt_oss_mxfp4_expert_decode_matches_dense_reference() {
    for_each_non_cpu_backend!(|B| {
        let context = create_context::<B>();
        let input = bf16_from_le_bytes(&INPUT_BF16);
        let checkpoint_up_biases = bf16_from_le_bytes(&GATE_UP_BIASES_CHECKPOINT);
        let down_biases = bf16_from_le_bytes(&DOWN_BIASES_BF16);
        let expected = dense_reference(
            &input,
            &GATE_UP_BLOCKS_CHECKPOINT,
            &GATE_UP_SCALES_CHECKPOINT,
            &DOWN_BLOCKS,
            &DOWN_SCALES,
            0.5,
            2.0,
            &checkpoint_up_biases,
            &down_biases,
            D_MODEL,
            D_FF,
        );
        let (w13_blocks, w13_scales, up_biases) = stack_gate_up_mxfp4(
            &GATE_UP_BLOCKS_CHECKPOINT,
            &GATE_UP_SCALES_CHECKPOINT,
            &checkpoint_up_biases,
            D_MODEL,
            D_FF,
        );

        assert_eq!(w13_blocks.len(), 2 * D_FF * (D_MODEL / 16) * 8);
        assert_eq!(w13_scales.len(), 2 * D_FF * (D_MODEL / 16));
        assert_eq!(DOWN_BLOCKS.len(), D_MODEL * D_FF.div_ceil(32) * 16);

        let input = alloc_allocation_with_data::<B, bf16>(&context, &input);
        let offsets = alloc_allocation_with_data::<B, u32>(&context, &[0, 1]);
        let w13_blocks = alloc_allocation_with_data::<B, u8>(&context, &w13_blocks);
        let w13_scales = alloc_allocation_with_data::<B, u8>(&context, &w13_scales);
        let w13_global_scale = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::from_f32(0.5)]);
        let w2_blocks = alloc_allocation_with_data::<B, u8>(&context, &DOWN_BLOCKS);
        let w2_scales = alloc_allocation_with_data::<B, u8>(&context, &DOWN_SCALES);
        let w2_global_scale = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::from_f32(2.0)]);
        let up_biases = alloc_allocation_with_data::<B, bf16>(&context, &up_biases);
        let down_biases = alloc_allocation_with_data::<B, bf16>(&context, &down_biases);

        let block = MoeExpertsMxfp4DecodeBlock::<B>::new(&context, DataType::BF16, 2).expect("MXFP4 decode block");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        let output = block
            .encode(
                MoeExpertsMxfp4Arguments {
                    x_perm: &input,
                    expert_offsets: &offsets,
                    w13_blocks: &w13_blocks,
                    w13_scales: &w13_scales,
                    w13_global_scale: &w13_global_scale,
                    w2_blocks: &w2_blocks,
                    w2_scales: &w2_scales,
                    w2_global_scale: &w2_global_scale,
                    up_biases: &up_biases,
                    down_biases: &down_biases,
                    total_rows: 1,
                    d_model: D_MODEL,
                    d_ff: D_FF,
                    num_routed_experts: 1,
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: 7.0,
                    up_clip_min: -6.0,
                    up_clip_max: 8.0,
                    silu_alpha: 1.702,
                },
                &mut encoder,
            )
            .expect("encode packed expert");

        let completed = encoder.end_encoding().submit().wait_until_completed().expect("run packed expert");
        let actual = allocation_prefix_to_vec::<B, bf16>(&output, D_MODEL);
        assert_eq_float(&expected, &actual, 0.02, "real GPT-OSS MXFP4 expert output");
        drop(output);
        drop(completed);
    });
}

#[uzu_test]
fn test_real_gpt_oss_multigroup_mxfp4_expert_decode_matches_dense_reference() {
    const MULTIGROUP_D_MODEL: usize = 64;
    const MULTIGROUP_D_FF: usize = 32;
    const PREFILL_ROWS: usize = 33;

    for_each_non_cpu_backend!(|B| {
        let context = create_context::<B>();
        let input_bytes = bytes_from_hex(MULTIGROUP_INPUT_BF16_HEX);
        let w13_blocks = bytes_from_hex(MULTIGROUP_GATE_UP_BLOCKS_HEX);
        let w13_scales = bytes_from_hex(MULTIGROUP_GATE_UP_SCALES_HEX);
        let checkpoint_up_bias_bytes = bytes_from_hex(MULTIGROUP_GATE_UP_BIASES_BF16_HEX);
        let w2_blocks = bytes_from_hex(MULTIGROUP_DOWN_BLOCKS_HEX);
        let w2_scales = bytes_from_hex(MULTIGROUP_DOWN_SCALES_HEX);
        let down_bias_bytes = bytes_from_hex(MULTIGROUP_DOWN_BIASES_BF16_HEX);

        let input = bf16_from_le_bytes(&input_bytes);
        let checkpoint_up_biases = bf16_from_le_bytes(&checkpoint_up_bias_bytes);
        let down_biases = bf16_from_le_bytes(&down_bias_bytes);
        let expected = dense_reference(
            &input,
            &w13_blocks,
            &w13_scales,
            &w2_blocks,
            &w2_scales,
            1.0,
            1.0,
            &checkpoint_up_biases,
            &down_biases,
            MULTIGROUP_D_MODEL,
            MULTIGROUP_D_FF,
        );
        // Repeat the captured checkpoint row once beyond the scheduler boundary.
        let prefill_input = input.repeat(PREFILL_ROWS);
        let prefill_expected = expected.repeat(PREFILL_ROWS);
        let (w13_blocks, w13_scales, up_biases) =
            stack_gate_up_mxfp4(&w13_blocks, &w13_scales, &checkpoint_up_biases, MULTIGROUP_D_MODEL, MULTIGROUP_D_FF);

        // These dimensions force four group-16 w13 groups and one complete group-32 w2 group.
        assert_eq!(w13_blocks.len(), 2 * MULTIGROUP_D_FF * (MULTIGROUP_D_MODEL / 16) * 8);
        assert_eq!(w13_scales.len(), 2 * MULTIGROUP_D_FF * (MULTIGROUP_D_MODEL / 16));
        assert_eq!(w2_blocks.len(), MULTIGROUP_D_MODEL * (MULTIGROUP_D_FF / 32) * 16);
        assert_eq!(w2_scales.len(), MULTIGROUP_D_MODEL * (MULTIGROUP_D_FF / 32));

        let input = alloc_allocation_with_data::<B, bf16>(&context, &input);
        let offsets = alloc_allocation_with_data::<B, u32>(&context, &[0, 1]);
        let prefill_input = alloc_allocation_with_data::<B, bf16>(&context, &prefill_input);
        let prefill_offsets = alloc_allocation_with_data::<B, u32>(&context, &[0, PREFILL_ROWS as u32]);
        let w13_blocks = alloc_allocation_with_data::<B, u8>(&context, &w13_blocks);
        let w13_scales = alloc_allocation_with_data::<B, u8>(&context, &w13_scales);
        let w13_global_scale = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::ONE]);
        let w2_blocks = alloc_allocation_with_data::<B, u8>(&context, &w2_blocks);
        let w2_scales = alloc_allocation_with_data::<B, u8>(&context, &w2_scales);
        let w2_global_scale = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::ONE]);
        let up_biases = alloc_allocation_with_data::<B, bf16>(&context, &up_biases);
        let down_biases = alloc_allocation_with_data::<B, bf16>(&context, &down_biases);

        let block = MoeExpertsMxfp4DecodeBlock::<B>::new(&context, DataType::BF16, 2).expect("MXFP4 decode block");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        let output = block
            .encode(
                MoeExpertsMxfp4Arguments {
                    x_perm: &input,
                    expert_offsets: &offsets,
                    w13_blocks: &w13_blocks,
                    w13_scales: &w13_scales,
                    w13_global_scale: &w13_global_scale,
                    w2_blocks: &w2_blocks,
                    w2_scales: &w2_scales,
                    w2_global_scale: &w2_global_scale,
                    up_biases: &up_biases,
                    down_biases: &down_biases,
                    total_rows: 1,
                    d_model: MULTIGROUP_D_MODEL,
                    d_ff: MULTIGROUP_D_FF,
                    num_routed_experts: 1,
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: 7.0,
                    up_clip_min: -6.0,
                    up_clip_max: 8.0,
                    silu_alpha: 1.702,
                },
                &mut encoder,
            )
            .expect("encode packed multigroup expert");

        let completed = encoder.end_encoding().submit().wait_until_completed().expect("run packed multigroup expert");
        let actual = allocation_prefix_to_vec::<B, bf16>(&output, MULTIGROUP_D_MODEL);
        assert_eq_float(&expected, &actual, 0.02, "real GPT-OSS multigroup MXFP4 expert output");
        drop(output);
        drop(completed);

        let block = MoeExpertsMxfp4PrefillBlock::<B>::new(&context, DataType::BF16, 2).expect("MXFP4 prefill block");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        let output = block
            .encode(
                MoeExpertsMxfp4Arguments {
                    x_perm: &prefill_input,
                    expert_offsets: &prefill_offsets,
                    w13_blocks: &w13_blocks,
                    w13_scales: &w13_scales,
                    w13_global_scale: &w13_global_scale,
                    w2_blocks: &w2_blocks,
                    w2_scales: &w2_scales,
                    w2_global_scale: &w2_global_scale,
                    up_biases: &up_biases,
                    down_biases: &down_biases,
                    total_rows: PREFILL_ROWS,
                    d_model: MULTIGROUP_D_MODEL,
                    d_ff: MULTIGROUP_D_FF,
                    num_routed_experts: 1,
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: 7.0,
                    up_clip_min: -6.0,
                    up_clip_max: 8.0,
                    silu_alpha: 1.702,
                },
                &mut encoder,
            )
            .expect("encode packed multigroup prefill expert");

        let completed =
            encoder.end_encoding().submit().wait_until_completed().expect("run packed multigroup prefill expert");
        let actual = allocation_prefix_to_vec::<B, bf16>(&output, PREFILL_ROWS * MULTIGROUP_D_MODEL);
        assert_eq_float(&prefill_expected, &actual, 0.02, "real GPT-OSS multigroup MXFP4 prefill output");
        drop(output);
        drop(completed);
    });
}

#[uzu_test]
#[ignore]
fn test_gpt_oss_mxfp4_expert_decode_perf() {
    for_each_non_cpu_backend!(|B| {
        const MODEL_DIM: usize = 2880;
        const EXPERT_HIDDEN_DIM: usize = 2880;

        let context = create_context::<B>();
        let model_groups = MODEL_DIM / 16;
        let hidden_groups = EXPERT_HIDDEN_DIM / 32;

        let input = alloc_allocation_with_data::<B, bf16>(&context, &vec![bf16::ZERO; MODEL_DIM]);
        let offsets = alloc_allocation_with_data::<B, u32>(&context, &[0, 1]);
        let w13_blocks =
            alloc_allocation_with_data::<B, u8>(&context, &vec![0; 2 * EXPERT_HIDDEN_DIM * model_groups * 8]);
        let w13_scales =
            alloc_allocation_with_data::<B, u8>(&context, &vec![127; 2 * EXPERT_HIDDEN_DIM * model_groups]);
        let w13_global_scale = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::ONE]);
        let w2_blocks = alloc_allocation_with_data::<B, u8>(&context, &vec![0; MODEL_DIM * hidden_groups * 16]);
        let w2_scales = alloc_allocation_with_data::<B, u8>(&context, &vec![127; MODEL_DIM * hidden_groups]);
        let w2_global_scale = alloc_allocation_with_data::<B, bf16>(&context, &[bf16::ONE]);
        let up_biases = alloc_allocation_with_data::<B, bf16>(&context, &vec![bf16::ZERO; 2 * EXPERT_HIDDEN_DIM]);
        let down_biases = alloc_allocation_with_data::<B, bf16>(&context, &vec![bf16::ZERO; MODEL_DIM]);

        const ENCODE_COUNT: usize = 100;
        let block = MoeExpertsMxfp4DecodeBlock::<B>::new(&context, DataType::BF16, 2).expect("MXFP4 block");
        let run = |block: &MoeExpertsMxfp4DecodeBlock<B>| {
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            for _ in 0..ENCODE_COUNT {
                let output = block
                    .encode(
                        MoeExpertsMxfp4Arguments {
                            x_perm: &input,
                            expert_offsets: &offsets,
                            w13_blocks: &w13_blocks,
                            w13_scales: &w13_scales,
                            w13_global_scale: &w13_global_scale,
                            w2_blocks: &w2_blocks,
                            w2_scales: &w2_scales,
                            w2_global_scale: &w2_global_scale,
                            up_biases: &up_biases,
                            down_biases: &down_biases,
                            total_rows: 1,
                            d_model: MODEL_DIM,
                            d_ff: EXPERT_HIDDEN_DIM,
                            num_routed_experts: 1,
                            gate_clip_min: f32::NEG_INFINITY,
                            gate_clip_max: 7.0,
                            up_clip_min: -6.0,
                            up_clip_max: 8.0,
                            silu_alpha: 1.702,
                        },
                        &mut encoder,
                    )
                    .expect("encode packed expert");
                drop(output);
            }
            let completed = encoder.end_encoding().submit().wait_until_completed().expect("run packed expert");
            completed.gpu_execution_time().as_secs_f64() * 1000.0 / ENCODE_COUNT as f64
        };

        for _ in 0..3 {
            run(&block);
        }
        let mut times = (0..20).map(|_| run(&block)).collect::<Vec<_>>();
        times.sort_by(|a, b| a.partial_cmp(b).expect("finite GPU time"));
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let median = times[times.len() / 2];
        let min = times[0];
        let max = times[times.len() - 1];

        eprintln!("  GPT-OSS MXFP4  mean={mean:.3}ms median={median:.3}ms min={min:.3}ms max={max:.3}ms");
    });
}
