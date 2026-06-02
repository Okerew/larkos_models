graph TD
Start([Training Start])-- > InitMeta["Initialize Meta State & LR"]
InitMeta-- > EpochLoop["Each Epoch"]
EpochLoop-- > ReadBackend["Read Backend State: Meta,Context, Neurons, Memory, Imagination, Emotions/Affective system etc."]
ReadBackend-- > DeriveParams["Derive Parameters: Alpha,<br/>Exploration Rate, Mem Weights"]
DeriveParams-- > SampleText["Sample Text Input from Data Pipeline"]
SampleText-- > InputPipeline["Input Pipeline: <br/> Normalize-->Fourier Encode --> Temporal Stack"]
InputPipeline-- > ForwardPass["_forward"]
ForwardPass-- > MCDropout["MC Dropout Probing<br/>Variance Estimation"]
MCDropout-- > ModelPred["Model Prediction LarkosModel"]
ModelPred-- > CrossAttn["Cross Attention<br/>+ Text Projection"]
CrossAttn-- > CognitiveFuse["Cognitive Fusion<br/>C-Side Bridge"]
CognitiveFuse-- > FusionTransformer["Fusion Transformer<br/>Band Attention"]
FusionTransformer-- > MAMLInner["MAML Inner Loop<br/>Adapted Model"]
MAMLInner-- > FwdResult["Forward Result: model_pred,<br/>fused, maml_pred, target"]
FwdResult-- > BackwardPass["_backward"]
BackwardPass-- > ComputeLoss["Compute Loss: outer + base + pred + aux_loss"]
ComputeLoss-- > GradientBalance["Gradient Balancing<br/>Rescale Embed Weights"]
GradientBalance-- > GradClip["Clip Gradients<br/>Global Norm = 1.0"]
GradClip-- > OptimStep["Optimizer Step +<br/>Scheduler Step + EMA Update"]
OptimStep-- > LossVal["Return loss_val"]
LossVal-- > SideEffects["_side_effects"]
SideEffects-- > TextDecode["Text Decode Readout Only"]
TextDecode-- > BackendUpdate["Backend Updates: Memory,<br/>Neurons, Motivation, Identity"]
BackendUpdate-- > Emotions["Emotional Pipeline: Trigger<br/>Emotions, Update Bonds"]
Emotions-- > Reflection["Reflection Metrics: Confidence,<br/>Drift, Novelty, Coherence"]
Reflection-- > Verification["Verification Check<br/>Pattern Tracking"]
Verification-- > Logging["Log Metrics to File"]
Logging-- > TargetRefresh{Target Freeze < br /> Window Over ?}
TargetRefresh-- >| Yes | RefreshTarget["Refresh Cached Target<br/>Reset Fused Cache"]
TargetRefresh-- >| No | KeepFrozen["Keep Frozen Target & Input"]
RefreshTarget-- > EpochEnd{More Epochs ?}
KeepFrozen-- > EpochEnd
EpochEnd-- >| Yes | EpochLoop
EpochEnd-- >| No | PostTrain["Post-Training"]
PostTrain-- > ConsolidateMem["Consolidate Memory"]
ConsolidateMem-- > SaveCheckpoint["Save Checkpoint<br/>larkos_model.pt"]
SaveCheckpoint-- > SaveNetwork["Save Network States & Memory"]
SaveNetwork-- > FinalReflection["Get Identity Reflection"]
FinalReflection-- > End([Training Complete])
    style Start fill:#4CAF50, stroke:#2E7D32, color: #fff
    style End fill: #E91E63, stroke:#880E4F, color: #fff
    style ForwardPass fill:#2196F3, stroke:#1565C0, color: #fff
    style BackwardPass fill: #FF9800, stroke: #E65100, color: #fff
    style SideEffects fill:#9C27B0, stroke:#6A1B9A, color: #fff
    style EpochLoop fill: #FFC107, stroke: #F57F17, color:#000
