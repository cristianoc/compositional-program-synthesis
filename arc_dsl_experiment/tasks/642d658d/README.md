# ARC Task 642d658d: Universal Pattern Matching

## Task Overview

This ARC (Abstraction and Reasoning Corpus) task requires finding the output color for a given test input by learning patterns from training examples.

| Train 0 Input | Train 0 Output |
|---|---|
| ![](images/train_0_in.png) | ![](images/train_0_out.png) |

| Train 1 Input | Train 1 Output |
|---|---|
| ![](images/train_1_in.png) | ![](images/train_1_out.png) |

| Train 2 Input | Train 2 Output |
|---|---|
| ![](images/train_2_in.png) | ![](images/train_2_out.png) |

| Test 0 Input |
|---|
| ![](images/test_0_in.png) |

**The Goal:** Given the test input, predict the output color (a single color for the entire grid).

## Core Concepts

### 1. Universal Schemas (Patterns)

A **universal schema** is a pattern template that captures consistent structures across training examples. It uses three types of symbols:

- **Numbers (0-9)**: Must match exactly this color
- **Variables (X, Y, Z)**: Must be the same color within the pattern, but can be any color
- **Wildcards (*)**: Can be any color, no constraints

#### Example Schema (3×3):
```
[4 X *]
[X * *] 
[Y * Y]
```

This pattern means:
- Top-left must be color 4 (yellow)
- Position (0,1) and (1,0) must be the same color (call it X)
- Position (2,0) and (2,2) must be the same color (call it Y)
- All * positions can be any color

### 2. Pattern Matching Process

For each training example, we:
1. **Extract windows** of different shapes (1×3, 3×1, 2×3, 3×3, 5×5) from the input grid
2. **For each position in the window template**, collect windows where that position contains color 4
3. **Create schemas** from these window collections by identifying constant vs. variable positions
4. **Intersect schemas** across all examples to find universal patterns

#### Window Collection Example:
For 3×3 windows and position (1,1) = center:
```
All 3×3 windows from grid:

Window 1:    Window 2:    Window 3:
[1 0 3]      [0 3 2]      [3 2 1]
[0 2 4]      [2 4 1]      [4 1 3]
[3 1 4]      [1 4 0]      [4 0 2]

Keep only windows where position (1,1) = 4:

Window 2:
[0 3 2]  ← center position (1,1) contains 4
[2 4 1]  
[1 4 0]  

For position (0,0) = top-left, keep only windows where (0,0) = 4:

Window 3:
[4 1 3]  ← top-left position (0,0) contains 4
[0 2 4]
[2 4 1]
```

### 3. Schema Intersection

For each window position where we collected windows with color 4, we find the **intersection** - the pattern that holds across ALL training examples.

#### Example Intersection:
For position (1,1) - windows where center = 4:
```
Training example 1:
[0 3 2]    
[2 4 1]  ← center is 4
[1 4 0]

Training example 2:
[1 0 5]    
[3 4 2]  ← center is 4  
[0 1 3]

Training example 3:
[2 1 0]
[5 4 0]  ← center is 4
[3 2 1]

Intersection schema:
[* * *]    (no consistent pattern except...)
[* 4 *]    (center is always 4 - as required)
[* * *]    (no other positions are constant)
```

The intersection keeps only the constraints that hold across ALL examples. Since we collected windows where position (1,1) = 4, that position will always be 4 in the final schema.

### 3.5. Pattern Position Selection

For each shape, multiple schema positions are generated (e.g., 9 positions for 3×3). The system automatically selects the **most structurally complex** position using this scoring:

**Structure Score = Constraint Positions + (2 × Variable Relationships) + Variable Diversity**

Where:
- **Constraint Positions**: Number of non-wildcard positions (numbers + variables)
- **Variable Relationships**: Extra occurrences of variables (X appearing 4 times = 3 relationships)  
- **Variable Diversity**: Number of different variables (X, Y, Z = 3)

#### Example: (3,3) Position Selection
```
Position (1,1): [*, X, *][X, 4, X][*, X, *]  → Score: 12 (5 constraints, 3 relationships, 1 var)
Position (0,0): [4, X, *][X, *, *][Y, *, Y] → Score: 11 (5 constraints, 2 relationships, 2 vars)  
Position (0,1): [X, 4, X][*, X, *][*, *, *] → Score: 9  (4 constraints, 2 relationships, 1 var)
```

**Winner: Position (1,1)** - The perfect cross pattern with maximum variable relationships.

### 4. Universal Matching

The universal matcher `match_universal_pos(shape=(h,w))` works as follows:

1. **Input**: A grid and pre-computed universal schemas for shape (h,w)
2. **Process**: Slide the schemas across every possible position in the grid
3. **Output**: All positions where a schema matches, plus what color should go there

#### Matching Example:
```
Grid:           Schema:        Match at (0,0)?
[0 4 2]        [4 * *]        
[1 4 3]   vs   [* 4 *]   →   Check positions:
[2 0 1]        [* * 4]        (0,0): 0 vs 4 ✗

Grid:           Schema:        Match at (0,1)?  
[0 4 2]        [4 * *]
[1 4 3]   vs   [* 4 *]   →   Extract 3x3 starting at (0,1):
[2 0 1]        [* * 4]        [4 2 ?] - window goes outside grid ✗

Match requires the full schema to fit within grid bounds.
```

### 5. Aggregation: From Matches to Color

After finding all pattern matches from the selected optimal position, we need to decide on a single output color. Different **aggregators** use different strategies:

**Important**: Aggregators only consider positions with actual constraints (numbers or variables). **Wildcard positions ("*") are completely ignored** during color aggregation.

#### Example: Aggregation with Wildcards
```
Schema:    Matched Window:    Colors for Aggregation:
[4 X *]  →  [4 2 7]        →  [4, 2] (ignores 7 from "*" position)
[X * *]     [2 9 1]           [2] (ignores 9,1 from "*" positions)  
[Y * Y]     [3 0 3]           [3, 3] (ignores 0 from "*" position)

Final aggregation: [4, 2, 2, 3, 3] → mode = 2 or 3 (tie-breaking applies)
```

#### OpUniformColorFromMatches
- Collect all colors from constraint positions (ignore "*" positions)
- Return the most common (mode) color

#### OpUniformColorFromMatchesExcludeGlobal  
- Same as above, but ignore the most common color in the entire grid
- Useful when the background color dominates

#### OpUniformColorFromMatchesUniformNeighborhood
- For each match, check if the neighborhood around the center has uniform color
- Only use matches where the neighborhood is uniform
- Return mode of those colors

### 6. Program Pipeline

A complete program is a pipeline: **Matcher → Aggregator**

Example: `match_universal_pos(shape=(3,3)) |> OpUniformColorFromMatchesUniformNeighborhood`

1. **Matcher** finds all 3×3 positions where universal schemas match
2. **Aggregator** filters to uniform neighborhoods and returns mode color

## Method: Universal Pattern Learning

### Training Phase
1. **Extract windows** around color 4 centers from all training inputs
2. **Build universal schemas** by intersecting patterns across examples  
3. **Select best pattern position** for each shape based on structural complexity
4. **Test multiple shapes** (1×3, 3×1, 2×3, 3×3, 5×5) using their optimal patterns - only shapes achieving perfect training accuracy produce programs

### Prediction Phase  
1. **Apply universal matchers** to find all pattern occurrences in test input
2. **Use aggregators** to convert matches into a single predicted color
3. **Return programs** that work perfectly on training data

## Results

### Found Programs
The system discovers programs like:
- `match_universal_pos(shape=(1, 3),pos=(0, 1)) |> OpUniformColorFromMatchesUniformNeighborhood [✓ 1/1 test]`
- `match_universal_pos(shape=(2, 3),pos=(0, 1)) |> OpUniformColorFromMatchesExcludeGlobal(cross_only=True) [✓ 1/1 test]`
- `match_universal_pos(shape=(3, 3),pos=(1, 1)) |> OpUniformColorFromMatchesExcludeGlobal(cross_only=True) [✓ 1/1 test]`

The `[✓ 1/1 test]` indicator shows this program correctly predicts the test case.

### Visual Results

![](images/overlay_mosaic.png)

This mosaic shows:
- **Left panels**: Input grids with yellow rectangles marking where the optimal patterns match
- **Right panels**: The predicted output color from the best aggregator for each shape
- **Columns**: Successful pattern shapes (1×3, 3×1, 2×3, 3×3) using their structurally best positions
- **Rows**: Training and test examples

**Pattern Selection**: Yellow rectangles show matches for the structurally most complex pattern position of each **successful** shape:
- (1×3): Position (0,1) - Linear symmetry `[X, 4, X]`
- (2×3): Position (0,1) - T-pattern `[X, 4, X][*, X, *]`  
- (3×1): Position (1,0) - Vertical symmetry
- (3×3): Position (1,1) - Perfect cross pattern

**Note**: (5×5) patterns are generated but don't produce programs that achieve perfect training accuracy, so they don't appear in the found programs list. This demonstrates that **structural complexity doesn't guarantee predictive success** - patterns can be highly structured but still fail to generalize.

### Important Finding: Aggregator Sensitivity

The 5×5 results reveal a critical insight about the interaction between pattern matching and aggregation:

```
- match_universal_pos(shape=(5, 5)) |> OpUniformColorFromMatches [✗ 0/1 test]
- match_universal_pos(shape=(5, 5)) |> OpUniformColorFromMatchesExcludeGlobal [✗ 0/1 test]  
- match_universal_pos(shape=(5, 5)) |> OpUniformColorFromMatchesExcludeGlobal(cross_only=True) [✓ 1/1 test]
```

**The same pattern matcher** `match_universal_pos(shape=(5, 5))` finds valid, structured patterns, but different aggregators interpret these matches differently. This reveals several insights:

**What this means for finding the "right" solution:**

1. **Pattern quality ≠ Aggregation success**: Valid patterns can fail with wrong aggregators
2. **Aggregator choice is crucial**: The same matches interpreted differently can succeed or fail
3. **Training accuracy is necessary but not sufficient**: All these programs work on training, but aggregation strategy determines test success
4. **Cross-only filtering helps**: The successful variant uses `cross_only=True`, suggesting spatial filtering improves generalization

**Visual impact**: The mosaic uses the first found aggregator for each shape, so the 5×5 column shows results from `OpUniformColorFromMatches` (which fails test), not `OpUniformColorFromMatchesExcludeGlobal(cross_only=True)` (which succeeds).

This demonstrates that **the "right" solution depends on both finding good patterns AND choosing the right way to aggregate the results**. Test performance indicators (✓/✗) are essential for identifying which aggregation strategies generalize beyond training data.

## Technical Implementation

### Core Operations

**Grid → Matches**: `match_universal_pos(shape=(h,w))`
- Finds all positions where universal schemas of size h×w match
- Returns match locations and associated colors

**Matches → Color**: Various aggregators
- `OpUniformColorFromMatches`: Simple mode of matched colors
- `OpUniformColorFromMatchesExcludeGlobal`: Mode excluding background  
- `OpUniformColorFromMatchesUniformNeighborhood`: Mode of uniform neighborhoods
- `OpUniformColorPerSchemaThenMode`: Per-schema mode, then global mode

### Search Strategy

The system enumerates all possible **Matcher + Aggregator** combinations and keeps those that achieve perfect accuracy on training examples. Test performance is evaluated separately and clearly marked.

## Running the Code

### Requirements
- Python 3.10+ with `numpy`

### Usage
```bash
python3 repro.py
```

### Outputs
- **Visual**: `images/overlay_mosaic.png` - Pattern matches and predictions
- **Programs**: `programs_found.json` - All discovered programs with test results  
- **Stats**: `repro_stats.json` - Performance metrics and timing

### Customization
To try different pattern shapes:
```python
dsl.enumerate_programs_for_task(task, universal_shapes=[(1,3),(3,1),(2,3),(3,3),(5,5)])
```

## Key Insights

1. **Universal patterns** work across all training examples, making them robust
2. **Structural complexity selection** automatically identifies the most informative pattern position for each shape, eliminating the need for complex post-processing filters
3. **Quality through selection, not filtering**: Choosing the right pattern position is more effective than applying complex match filtering
4. **Multiple shapes** capture different types of spatial relationships at optimal positions
5. **Test indicators** (✓/✗) help distinguish reliable vs. overfitted programs
6. **Aggregation strategy** is crucial - different tasks need different approaches
7. **Cross patterns emerge naturally**: Center positions (1,1) consistently yield the highest structural scores across shapes
8. **Simple visualization**: All matches from optimal patterns provide clean, interpretable results without additional filtering
9. **Natural selection pressure**: Not all pattern shapes produce working programs - only those achieving perfect training accuracy survive the enumeration process

The core insight is that ARC tasks often have consistent local patterns, and by finding the intersection of these patterns across examples, we can build reliable predictors for new inputs. **Structural complexity-based pattern selection** ensures we focus on the most informative patterns without complex post-processing. However, **perfect training accuracy does not guarantee the right solution** - test evaluation reveals which aggregation strategies truly generalize beyond the training data.