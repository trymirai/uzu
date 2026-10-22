use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uzu::session::tool::{func_def::ErrorFuture, uzu_tool_function};

/// Data for an inline chart. Only the fields in this schema are accepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChartSpec {
    #[serde(rename = "type")]
    pub kind: ChartType,
    /// A concise chart title, at most 200 characters.
    #[schemars(length(min = 1, max = 200))]
    pub title: String,
    /// Chart height: small (320 px), medium (480 px, the default), or big (640 px).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<ChartHeight>,
    /// Category names, required for every chart except scatter and bubble.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1, max = 500))]
    pub labels: Option<Vec<String>>,
    /// One to eight series, with at most 2,000 data points in the whole chart.
    #[schemars(length(min = 1, max = 8))]
    pub datasets: Vec<ChartDataset>,
    /// Optional horizontal axis label, at most 200 characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1, max = 200))]
    pub x_label: Option<String>,
    /// Optional vertical axis label, at most 200 characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1, max = 200))]
    pub y_label: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ChartType {
    Bar,
    Line,
    Scatter,
    Bubble,
    Pie,
    Doughnut,
    Radar,
    PolarArea,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ChartHeight {
    Small,
    Medium,
    Big,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChartDataset {
    /// A concise series name, at most 200 characters.
    #[schemars(length(min = 1, max = 200))]
    pub label: String,
    /// Numbers matching labels for categorical charts, or x/y points for scatter and bubble.
    #[schemars(length(min = 1, max = 500))]
    pub data: Vec<ChartDatum>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ChartDatum {
    Number(f64),
    Point(ChartPoint),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChartPoint {
    pub x: f64,
    pub y: f64,
    /// Bubble radius in pixels, from 0 to 100. Required for bubble, omitted for scatter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0, max = 100))]
    pub r: Option<f64>,
}

impl ChartSpec {
    pub(super) fn validate(&self) -> Result<(), ErrorFuture> {
        validate_text(&self.title)?;
        for text in self.x_label.iter().chain(self.y_label.iter()) {
            validate_text(text)?;
        }
        if !(1..=8).contains(&self.datasets.len()) {
            return Err("A chart must have 1 to 8 datasets".into());
        }
        let point_chart = matches!(self.kind, ChartType::Scatter | ChartType::Bubble);
        if point_chart {
            if self.labels.is_some() {
                return Err("Scatter and bubble charts use x/y points, not category labels".into());
            }
        } else {
            let labels = self.labels.as_ref().ok_or("This chart type requires category labels")?;
            if !(1..=500).contains(&labels.len()) {
                return Err("A chart must have 1 to 500 category labels".into());
            }
            for label in labels {
                validate_text(label)?;
            }
        }
        let mut total_points = 0;
        for dataset in &self.datasets {
            validate_text(&dataset.label)?;
            if !(1..=500).contains(&dataset.data.len()) {
                return Err("Each dataset must have 1 to 500 data points".into());
            }
            total_points += dataset.data.len();
            if total_points > 2_000 {
                return Err("A chart must have at most 2,000 data points".into());
            }
            if let Some(labels) = &self.labels
                && dataset.data.len() != labels.len()
            {
                return Err("Each dataset must have one value per category label".into());
            }
            for datum in &dataset.data {
                match datum {
                    ChartDatum::Number(value) if !point_chart => {
                        if !value.is_finite() {
                            return Err("Chart values must be finite numbers".into());
                        }
                        if matches!(self.kind, ChartType::Pie | ChartType::Doughnut | ChartType::PolarArea)
                            && *value < 0.0
                        {
                            return Err("Pie, doughnut, and polar area values must be nonnegative".into());
                        }
                    },
                    ChartDatum::Point(point) if point_chart => {
                        if !point.x.is_finite() || !point.y.is_finite() {
                            return Err("Chart coordinates must be finite numbers".into());
                        }
                        match (self.kind, point.r) {
                            (ChartType::Scatter, None) => {},
                            (ChartType::Bubble, Some(r)) if r.is_finite() && (0.0..=100.0).contains(&r) => {},
                            _ => {
                                return Err(
                                    "Bubble points require radius r from 0 to 100; scatter points omit r".into()
                                );
                            },
                        }
                    },
                    _ => {
                        return Err(
                            "Use numeric data for categorical charts and x/y points for scatter or bubble".into()
                        );
                    },
                }
            }
        }
        Ok(())
    }
}

fn validate_text(text: &str) -> Result<(), ErrorFuture> {
    if text.trim().is_empty() || text.chars().count() > 200 {
        Err("Chart titles and labels must contain 1 to 200 characters".into())
    } else {
        Ok(())
    }
}

/// Display a chart inline in the conversation. Supply chart data only, never code, callbacks, plugins,
/// URLs, or Chart.js options. Use numeric values and matching labels for categorical charts; use x/y
/// points for scatter and x/y/r points for bubble. Optional height is small (320 px), medium
/// (480 px, default), or big (640 px). The returned chart is displayed to the user.
#[uzu_tool_function]
pub(super) fn show_chart(chart: ChartSpec) -> Result<ChartSpec, ErrorFuture> {
    chart.validate()?;
    Ok(chart)
}

#[cfg(test)]
mod tests {
    use uzu::session::tool::func_def::ToolDescriptor;

    use super::*;

    fn category_chart(kind: &str) -> serde_json::Value {
        serde_json::json!({
            "type": kind,
            "title": "Monthly sales",
            "labels": ["January", "February"],
            "datasets": [{ "label": "Sales", "data": [10, 20] }],
        })
    }

    #[tokio::test]
    async fn tool_returns_validated_data_for_every_chart_type() {
        let tool: ToolDescriptor = show_chart.into();
        for kind in ["bar", "line", "pie", "doughnut", "radar", "polarArea", "scatter", "bubble"] {
            let mut chart = category_chart(kind);
            if matches!(kind, "scatter" | "bubble") {
                chart.as_object_mut().unwrap().remove("labels");
                chart["datasets"][0]["data"] = if kind == "bubble" {
                    serde_json::json!([{ "x": 2, "y": 3, "r": 8 }])
                } else {
                    serde_json::json!([{ "x": 2, "y": 3 }])
                };
            }
            // Markup models pass object parameters as JSON text.
            for argument in [chart.clone(), serde_json::Value::String(chart.to_string())] {
                let result = tool.execute(serde_json::json!({ "chart": argument }).into()).await.unwrap();
                let returned: ChartSpec = serde_json::from_str(&result.json).unwrap();
                assert_eq!(returned, serde_json::from_value(chart.clone()).unwrap());
                returned.validate().unwrap();
                let json: serde_json::Value = serde_json::from_str(&result.json).unwrap();
                assert!(json.get("height").is_none());
                assert!(json.get("xLabel").is_none());
                assert!(json.get("yLabel").is_none());
                if kind == "scatter" {
                    assert!(json.get("labels").is_none());
                    assert!(json["datasets"][0]["data"][0].get("r").is_none());
                }
            }
        }
    }

    #[tokio::test]
    async fn tool_coerces_quoted_numeric_data_through_the_datum_union() {
        let tool: ToolDescriptor = show_chart.into();
        let mut chart = category_chart("bar");
        chart["datasets"][0]["data"] = serde_json::json!(["97.42", "123"]);
        for argument in [chart.clone(), serde_json::Value::String(chart.to_string())] {
            let result = tool.execute(serde_json::json!({ "chart": argument }).into()).await.unwrap();
            let returned: ChartSpec = serde_json::from_str(&result.json).unwrap();
            assert_eq!(returned.datasets[0].data, vec![ChartDatum::Number(97.42), ChartDatum::Number(123.0)]);
        }
    }

    #[tokio::test]
    async fn tool_coerces_point_coordinates_through_the_datum_union() {
        let tool: ToolDescriptor = show_chart.into();
        for kind in ["scatter", "bubble"] {
            let mut chart = category_chart(kind);
            chart.as_object_mut().unwrap().remove("labels");
            let mut point = serde_json::json!({ "x": "1", "y": "2" });
            let radius = if kind == "bubble" {
                point["r"] = serde_json::json!("8");
                Some(8.0)
            } else {
                None
            };
            chart["datasets"][0]["data"] = serde_json::json!([point]);
            for argument in [chart.clone(), serde_json::Value::String(chart.to_string())] {
                let result = tool.execute(serde_json::json!({ "chart": argument }).into()).await.unwrap();
                let returned: ChartSpec = serde_json::from_str(&result.json).unwrap();
                assert_eq!(
                    returned.datasets[0].data,
                    vec![ChartDatum::Point(ChartPoint {
                        x: 1.0,
                        y: 2.0,
                        r: radius,
                    })]
                );
            }
        }
    }

    #[tokio::test]
    async fn tool_preserves_supported_heights_and_rejects_arbitrary_sizes() {
        let tool: ToolDescriptor = show_chart.into();
        for height in ["small", "medium", "big"] {
            let mut chart = category_chart("bar");
            chart["height"] = serde_json::json!(height);
            for argument in [chart.clone(), serde_json::Value::String(chart.to_string())] {
                let result = tool.execute(serde_json::json!({ "chart": argument }).into()).await.unwrap();
                let returned: serde_json::Value = serde_json::from_str(&result.json).unwrap();
                assert_eq!(returned["height"], height);
                let chart: ChartSpec = serde_json::from_value(returned).unwrap();
                chart.validate().unwrap();
            }
        }
        for height in [serde_json::json!("large"), serde_json::json!("320px"), serde_json::json!(320)] {
            let mut chart = category_chart("bar");
            chart["height"] = height;
            assert!(tool.execute(serde_json::json!({ "chart": chart }).into()).await.is_err());
        }
    }

    #[tokio::test]
    async fn schema_and_execution_reject_options_code_and_unknown_nested_fields() {
        let tool: ToolDescriptor = show_chart.into();
        let schema: serde_json::Value = tool.parameters.clone().unwrap().try_into().unwrap();
        let chart_schema = &schema["properties"]["chart"];
        assert_eq!(chart_schema["additionalProperties"], false);
        assert_eq!(chart_schema["properties"]["datasets"]["maxItems"], 8);
        assert_eq!(chart_schema["properties"]["datasets"]["items"]["additionalProperties"], false);
        for key in ["options", "plugins", "code", "url", "onClick"] {
            let mut chart = category_chart("bar");
            chart[key] = serde_json::json!("alert('not executable')");
            assert!(tool.execute(serde_json::json!({ "chart": chart }).into()).await.is_err());
        }
        let mut chart = category_chart("bar");
        chart["datasets"][0]["backgroundColor"] = serde_json::json!("url(https://example.com/image)");
        assert!(tool.execute(serde_json::json!({ "chart": chart }).into()).await.is_err());
        let chart = serde_json::json!({
            "type": "scatter", "title": "Points", "datasets": [{
                "label": "Points", "data": [{ "x": 1, "y": 2, "callback": "alert(1)" }]
            }]
        });
        assert!(tool.execute(serde_json::json!({ "chart": chart }).into()).await.is_err());
    }

    #[test]
    fn validation_enforces_shapes_sizes_and_finite_values() {
        let valid: ChartSpec = serde_json::from_value(category_chart("bar")).unwrap();
        let mut invalid = Vec::new();
        let mut chart = valid.clone();
        chart.title = " ".into();
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.datasets[0].label = "x".repeat(201);
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.labels = None;
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.datasets[0].data.pop();
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.datasets = vec![chart.datasets[0].clone(); 9];
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.labels = Some(vec!["Point".into(); 500]);
        chart.datasets = vec![
            ChartDataset {
                label: "Series".into(),
                data: vec![ChartDatum::Number(1.0); 500]
            };
            5
        ];
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.datasets[0].data[0] = ChartDatum::Number(f64::INFINITY);
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.kind = ChartType::Pie;
        chart.datasets[0].data[0] = ChartDatum::Number(-1.0);
        invalid.push(chart);
        let mut chart = valid.clone();
        chart.datasets[0].data[0] = ChartDatum::Point(ChartPoint {
            x: 1.0,
            y: 2.0,
            r: None,
        });
        invalid.push(chart);
        for chart in invalid {
            assert!(chart.validate().is_err(), "unexpectedly accepted {chart:?}");
        }
        for (kind, radius, accepted) in [
            (ChartType::Scatter, None, true),
            (ChartType::Scatter, Some(1.0), false),
            (ChartType::Bubble, None, false),
            (ChartType::Bubble, Some(0.0), true),
            (ChartType::Bubble, Some(100.0), true),
            (ChartType::Bubble, Some(101.0), false),
            (ChartType::Bubble, Some(f64::NAN), false),
        ] {
            let chart = ChartSpec {
                kind,
                labels: None,
                datasets: vec![ChartDataset {
                    label: "Points".into(),
                    data: vec![ChartDatum::Point(ChartPoint {
                        x: 1.0,
                        y: 2.0,
                        r: radius,
                    })],
                }],
                ..valid.clone()
            };
            assert_eq!(chart.validate().is_ok(), accepted);
        }
    }
}
