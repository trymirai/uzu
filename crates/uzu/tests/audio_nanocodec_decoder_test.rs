use uzu::audio::{
    AudioError,
    nanocodec::decoder::{NanoCodecDecoderGraph, NanoCodecDecoderJson},
};

fn decoder_json() -> serde_json::Value {
    serde_json::json!({
        "negative_slope": 0.01,
        "eps": 1e-9,
        "pre_conv": {
            "weight": {
                "shape": [2, 2, 1],
                "values": [1.0, 0.0, 0.0, 1.0]
            },
            "bias": [0.0, 0.0]
        },
        "stages": [
            {
                "activation_alpha": [1.0],
                "upsample_conv": {
                    "weight": {
                        "shape": [2, 1, 4],
                        "values": [
                            1.0, 0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0
                        ]
                    },
                    "bias": [0.0, 0.0],
                    "stride": 2,
                    "groups": 2
                }
            }
        ],
        "post_activation_alpha": [1.0],
        "post_conv": {
            "weight": {
                "shape": [1, 2, 1],
                "values": [1.0, 0.0]
            },
            "bias": [0.0]
        }
    })
}

fn parse_graph(value: serde_json::Value) -> NanoCodecDecoderGraph {
    let parsed: NanoCodecDecoderJson = serde_json::from_value(value).expect("json");
    NanoCodecDecoderGraph::try_from(parsed).expect("graph")
}

#[test]
fn decoder_graph_parses_and_reports_upsample_factor() {
    let graph = parse_graph(decoder_json());
    assert_eq!(graph.upsample_factor(), 2);
    assert_eq!(graph.output_channels(), 1);
}

#[test]
fn decoder_graph_decode_scales_lengths() {
    let graph = parse_graph(decoder_json());

    let batch_size = 1usize;
    let channels = 2usize;
    let frames = 3usize;
    let latent = vec![0.2, 0.1, -0.2, 0.0, 0.0, 0.0];
    let result = graph.decode_padded(&latent, &[2usize], batch_size, channels, frames).expect("decode");

    assert_eq!(result.channels, 1);
    assert_eq!(result.frames, 6);
    assert_eq!(result.lengths, vec![4usize]);
    assert_eq!(result.samples.len(), batch_size * result.channels * result.frames);
}

#[test]
fn decoder_graph_rejects_residual_channel_mismatch() {
    let mut json = decoder_json();
    json["stages"][0]["res_layer"] = serde_json::json!({
        "res_blocks": [{
            "res_blocks": [{
                "input_activation_alpha": [1.0],
                "input_conv": {
                    "weight": {
                        "shape": [2, 1, 1],
                        "values": [1.0, 0.0]
                    },
                    "bias": [0.0, 0.0]
                },
                "skip_activation_alpha": [1.0],
                "skip_conv": {
                    "weight": {
                        "shape": [2, 2, 1],
                        "values": [1.0, 0.0, 0.0, 1.0]
                    },
                    "bias": [0.0, 0.0]
                }
            }]
        }]
    });

    let parsed: NanoCodecDecoderJson = serde_json::from_value(json).expect("json");
    let error = NanoCodecDecoderGraph::try_from(parsed).expect_err("must reject mismatch");
    match error {
        AudioError::Runtime(message) => assert!(message.contains("residual input_conv channel mismatch")),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn decoder_graph_residual_layer_changes_output() {
    let without_residual = parse_graph(decoder_json());
    let mut with_residual_json = decoder_json();
    with_residual_json["stages"][0]["res_layer"] = serde_json::json!({
        "res_blocks": [{
            "res_blocks": [{
                "input_activation_alpha": [1.0],
                "input_conv": {
                    "weight": {
                        "shape": [2, 2, 1],
                        "values": [0.0, 0.0, 0.0, 0.0]
                    },
                    "bias": [0.0, 0.0]
                },
                "skip_activation_alpha": [1.0],
                "skip_conv": {
                    "weight": {
                        "shape": [2, 2, 1],
                        "values": [0.0, 0.0, 0.0, 0.0]
                    },
                    "bias": [0.5, 0.0]
                }
            }]
        }]
    });
    let with_residual = parse_graph(with_residual_json);

    let batch_size = 1usize;
    let channels = 2usize;
    let frames = 3usize;
    let lengths = [2usize];
    let latent = vec![0.2, 0.1, -0.2, 0.0, 0.0, 0.0];

    let baseline = without_residual.decode_padded(&latent, &lengths, batch_size, channels, frames).expect("decode");
    let with_residual = with_residual.decode_padded(&latent, &lengths, batch_size, channels, frames).expect("decode");

    assert_eq!(with_residual.channels, baseline.channels);
    assert_eq!(with_residual.frames, baseline.frames);
    assert_eq!(with_residual.lengths, baseline.lengths);

    let max_delta = with_residual
        .samples
        .iter()
        .zip(baseline.samples.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_delta > 1e-5, "residual layer should alter decoded waveform");
}
