# SmolVLA LeKiwi mixed-action redesign

## Summary

This document defines a SmolVLA training redesign for the `lekiwi_banana_and_blue_block_merged` dataset.

The redesign intentionally stops training the first 6 arm dimensions for now, restores them at inference time by directly passing through the current `observation.state`, and converts the mobile base `x.vel`, `y.vel`, and `theta.vel` into three categorical outputs. The external robot and evaluation interface remains a 9-dimensional action vector.

## Problem statement

The current LeKiwi dataset and task setup do not match the default SmolVLA action-learning assumptions well enough:

1. The first 6 action dimensions correspond to arm joint angles, but the current dataset focus is language-conditioned base navigation rather than arm control.
2. The base action dimensions are effectively discrete control values and are better modeled as classification than continuous regression.
3. The current evaluation and robot control path expects a full 9-dimensional action vector, so changing the internal training representation must not break deployment.
4. The near-term goal is to improve language-conditioned target-seeking behavior for the mobile base, not to preserve arm-action learning quality in this dataset.

## Goals

1. Remove arm-action loss from the current training objective without breaking the 9-dimensional runtime interface.
2. Train `x.vel`, `y.vel`, and `theta.vel` as three-way categorical outputs.
3. Keep the design compatible with later restoration of arm training.
4. Make the internal representation explicit so preprocessing, model outputs, postprocessing, and evaluation all agree.

## Non-goals

1. This design does not attempt to improve language grounding by itself.
2. This design does not change the robot hardware API.
3. This design does not enable useful `theta.vel` learning from the current merged dataset if the dataset still contains only zeros in that dimension.
4. This design does not convert the whole SmolVLA model into an ACT-style pure split-head policy.

## Chosen approach

Use a mixed internal action representation:

1. **Arm passthrough**
   - Do not train the first 6 dimensions.
   - At inference and evaluation time, recover those 6 dimensions by directly copying the current `observation.state[:6]`.

2. **Discrete base heads**
   - Replace the current continuous training target for `x.vel`, `y.vel`, and `theta.vel` with three categorical heads.
   - Each head predicts one of three discrete values, initially `[-0.25, 0.0, 0.25]`.

3. **Stable external interface**
   - Internally, the policy reasons about `arm_passthrough + base_discrete`.
   - Externally, the postprocessor always exports a full 9-dimensional action vector:
     - dims `0..5`: copied from current state
     - dims `6..8`: decoded from categorical predictions

This is the preferred option because it matches the current dataset semantics, reduces meaningless arm loss, and preserves a clean upgrade path for future arm reintroduction.

## Alternatives considered

### Option A: Keep 9-dimensional output and zero arm loss

Mask the first 6 action dimensions out of loss computation but still keep them in the model output tensor.

**Pros**
- Smaller code diff.
- Fewer pipeline changes.

**Cons**
- Internal semantics stay muddled because the model still appears to predict 9 learned dimensions.
- Easier to accidentally reintroduce arm loss or normalization mismatches.
- Harder to evolve into a clear mixed-action architecture later.

### Option C: Keep full 9-dimensional continuous flow output and add discrete base heads on top

Use continuous flow for all 9 dimensions but ignore or downweight the first 6 dimensions while also adding base auxiliary heads.

**Pros**
- Maximum continuity with current SmolVLA structure.

**Cons**
- The arm branch remains conceptually active even though it should not learn.
- Base semantics are duplicated between continuous flow output and categorical heads.
- Harder to reason about exported actions and debugging.

## Architecture

### Internal action contract

The policy will no longer treat the whole 9-dimensional robot action as a single homogeneous learned tensor.

Instead:

1. **Observed arm state**
   - Source: `observation.state[:6]`
   - Role: passthrough-only runtime payload
   - Training: excluded from model loss

2. **Discrete base targets**
   - `base_x_target`
   - `base_y_target`
   - `base_theta_target`
   - Role: categorical labels derived from raw action values

3. **Exported action**
   - Full 9-dimensional tensor assembled after inference

### Model structure

The SmolVLA body remains a language- and vision-conditioned backbone, but its action output layer is redefined:

1. Remove learned dependence on the first 6 exported action dimensions.
2. Add three classification heads on top of the final action hidden states:
   - `base_x_head`
   - `base_y_head`
   - `base_theta_head`
3. Decode logits to discrete values only in postprocessing/export, not inside the training target extraction step.

The design deliberately borrows the **explicit action-head configuration and target extraction ideas from ACT**, but applies them to SmolVLA as a classification-based base controller rather than as a full action-chunk split-head replacement.

## Detailed component changes

### 1. Configuration changes

File:

- `src/lerobot/policies/smolvla/configuration_smolvla.py`

Add explicit configuration for mixed-action training:

- `arm_passthrough_dims: list[int] = [0, 1, 2, 3, 4, 5]`
- `base_action_dims: list[int] = [6, 7, 8]`
- `use_discrete_base_heads: bool = True`
- `action_heads: list[SmolVLAActionHeadConfig]`
- `export_action_dim: int = 9`

Add a head config type modeled after ACT:

- `name`
- `type` (`categorical` only for this design)
- `index`
- `values`
- `loss_weight`

Initial defaults:

- `base_x`: values `[-0.25, 0.0, 0.25]`
- `base_y`: values `[-0.25, 0.0, 0.25]`
- `base_theta`: values `[-0.25, 0.0, 0.25]`

Validation rules:

1. Base head names must be unique.
2. Each base head must map exactly one exported action dimension.
3. `arm_passthrough_dims` and base head indices must not overlap.
4. `export_action_dim` must remain 9 for LeKiwi runtime compatibility.

### 2. Processor changes

File:

- `src/lerobot/policies/smolvla/processor_smolvla.py`

Add processor steps before normalization:

#### 2.1 `ExtractDiscreteBaseTargetsProcessorStep`

Responsibilities:

1. Read the raw 9-dimensional action.
2. Convert dims `6`, `7`, `8` into:
   - `base_x_target`
   - `base_y_target`
   - `base_theta_target`
3. Use nearest-value matching against the configured class values.

This mirrors the ACT target-extraction design, but only for the base dimensions.

#### 2.2 `ProjectSmolVLAActionForTrainingProcessorStep`

Responsibilities:

1. Preserve the original 9-dimensional action for export semantics if needed in complementary data.
2. Replace `TransitionKey.ACTION` with the reduced training representation used by the model.

For this design, the reduced training representation is **only the 3 base dimensions**, not the arm dimensions.

Rationale:

- The arm dimensions are passthrough-only and should not participate in loss.
- The model should only consume the dimensions it is asked to learn.

This means the model-facing action tensor for training becomes shape `(3,)`, representing raw base values before categorical conversion or, if we choose a pure classification path, may be omitted from the flow branch entirely.

#### 2.3 Recommended preprocessing order

1. Rename observations
2. Add batch dimension
3. Add newline to task
4. Tokenize task
5. Extract base categorical targets
6. Project action into model-facing training form
7. Normalize remaining model-facing features
8. Move to target device

### 3. Model changes

File:

- `src/lerobot/policies/smolvla/modeling_smolvla.py`

#### 3.1 Policy-level semantics

`SmolVLAPolicy` should stop treating exported action shape as the learned action shape.

Introduce two concepts:

1. **training action shape**
2. **export action shape**

The exported shape remains 9, but the learned action contract becomes base-only categorical prediction.

#### 3.2 VLAFlowMatching output redesign

Because the chosen design removes arm learning and makes the base categorical, the previous continuous action flow head is no longer the right fit for the current dataset-specific training target.

The recommended implementation is:

1. Keep the SmolVLA backbone and prefix/suffix conditioning path.
2. Replace the final continuous action projection for this mode with three categorical heads:
   - `base_x_head: hidden -> 3`
   - `base_y_head: hidden -> 3`
   - `base_theta_head: hidden -> 3`
3. Remove arm dimensions from the learned output tensor in this mode.

This effectively makes the current LeKiwi mode a **classification-style action decoder using the SmolVLA backbone**, not a full continuous flow decoder.

That is a deliberate design choice. It is justified because:

1. The current dataset’s learned action semantics are discrete for the base.
2. Arm learning is intentionally disabled.
3. Preserving continuous flow on unused arm dimensions provides no benefit.

#### 3.3 Forward pass

Training forward returns:

- `base_x_logits`
- `base_y_logits`
- `base_theta_logits`

Loss:

- `CE(base_x_logits, base_x_target)`
- `CE(base_y_logits, base_y_target)`
- `CE(base_theta_logits, base_theta_target)`

weighted by configured loss weights.

#### 3.4 Inference path

At inference:

1. Decode the three logits via `argmax`.
2. Map class ids back to real values.
3. Read `observation.state[:6]`.
4. Assemble exported action:

`[state[0], state[1], state[2], state[3], state[4], state[5], x_value, y_value, theta_value]`

### 4. Postprocessing changes

The postprocessor must own the final 9-dimensional export assembly.

Add a step such as:

#### `AssembleLeKiwiPassthroughActionProcessorStep`

Inputs:

- current observation state
- decoded base discrete action values

Output:

- full 9-dimensional action tensor

Responsibilities:

1. Copy arm dims from observation state.
2. Insert predicted base dims into positions `6..8`.
3. Preserve the action names expected by robot/evaluate processors.

## Evaluation path changes

File:

- `examples/lekiwi/evaluate.py`

Required changes:

1. Load SmolVLA instead of ACT.
2. Ensure the loaded postprocessor is the one that assembles full 9-dimensional exported actions.
3. Keep the robot-facing path unchanged after postprocessing.

The robot-facing code should continue to consume:

- `x.vel`
- `y.vel`
- `theta.vel`

No runtime consumer should need to know that arm dims are passthrough-only.

## Data and statistics handling

### Dataset stats

Normalization stats must match the model-facing representation.

For this mode:

1. Stats for arm action dimensions are irrelevant for learned targets.
2. Base categorical targets must not be normalized.
3. If a reduced action tensor is still passed through any normalization step, its stats must be projected to the same reduced dimensionality and ordering.

### Important caveat on `theta.vel`

If the current merged dataset still has `theta.vel == 0` everywhere, then:

1. the head should remain present for interface stability,
2. but either:
   - train it with zero weight, or
   - train it as a degenerate single-class distribution while documenting that it is temporarily inactive for real control learning.

The preferred behavior is **keep the head but set its loss weight to 0 until non-zero theta data exists**.

## Backward and forward compatibility

### Current compatibility target

This design is intentionally dataset-specific and should be gated by configuration.

Recommended config switch:

- `smolvla_mode = "lekiwi_base_discrete_passthrough"`

### Future recovery of arm learning

When arm supervision is reintroduced later:

1. remove arm passthrough-only mode,
2. restore continuous arm prediction,
3. keep base classification heads,
4. assemble output from:
   - learned arm continuous values
   - learned base discrete values

Because the current design separates passthrough and learned outputs explicitly, that transition will be straightforward.

## Testing strategy

Add tests for:

### Processor tests

1. raw 9-dimensional action maps to correct base class targets
2. arm dims are excluded from learned action representation
3. postprocessor assembles full 9-dimensional output correctly

### Model tests

1. forward returns base classification losses only in passthrough mode
2. decoded logits map back to configured discrete values
3. inference output arm dims equal current `observation.state[:6]`

### Evaluate tests

1. `examples/lekiwi/evaluate.py` loads SmolVLA correctly
2. robot-facing action still exposes 9 dimensions
3. base dims appear under expected keys

## Risks

1. **Dataset mismatch risk**
   - If arm motion becomes important again sooner than expected, passthrough mode will cap policy capability.

2. **Grounding risk**
   - This redesign improves action semantics, not target grounding.
   - If the core issue is language-vision shortcut learning, this alone will not fix it.

3. **Mode complexity**
   - Introducing a dataset-specific SmolVLA action mode increases configuration and test burden.

## Rollout plan

### Phase 1

Introduce processor and config changes, plus base categorical targets and 9-dimensional export assembly.

### Phase 2

Switch the LeKiwi SmolVLA mode to categorical-base training and validate on offline metrics and `evaluate.py`.

### Phase 3

Collect non-zero `theta.vel` data and enable real `base_theta` loss.

### Phase 4

Reintroduce arm learning when the dataset and task require it.

## Final recommendation

Implement a new SmolVLA LeKiwi mode where:

1. the first 6 action dimensions are **not trained**,
2. those 6 dimensions are **recovered from `observation.state` at runtime**,
3. `x.vel`, `y.vel`, and `theta.vel` are trained as **three separate categorical heads**,
4. the policy always exports a full **9-dimensional LeKiwi action vector**.

This design is the best fit for the current dataset and stated constraints because it aligns the model with the real control semantics while preserving a clean path back to future arm-action learning.
