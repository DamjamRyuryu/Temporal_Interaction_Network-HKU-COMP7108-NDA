# Temporal Interaction Network (TIN)

(Assignment of HKU-COMP7108)This project implements creation, analysis, and visualization of Temporal Interaction Networks (TIN). A TIN is a network where edges are annotated with the time of interaction between nodes and the quantity transferred.

## Features

The project includes the following features:

1. **TIN Creation**: Extract necessary fields from taxi data or load from pre-processed data
2. **Statistical Analysis**: Calculate network metrics such as number of nodes, edges, interactions, and average passengers per trip
3. **Data Provenance Tracking**: Track the origin of data quantities in the network using different strategies:
   - LRB (Least Recently Born): Prioritizes oldest data first
   - MRB (Most Recently Born): Prioritizes newest data first
4. **Temporal Analysis**: Analyze data flow at specific timestamps
5. **Network Visualization**: Visualize the TIN network structure with interactive edges and nodes

## Files Description

- `TIN.py`: Main program that implements the TIN class and its functionalities
- `preprocess_taxi_data.py`: Data preprocessing script for raw taxi data
- `requirements.txt`: List of project dependencies
- `yellow_taxi.csv`: Original data file (800MB+, not included in repository*)
- `tin_graph.txt`: Processed TIN data file (automatically generated, limited to 1 million edges)
- `graph.txt`: Small test file for development and testing


*: Might be available at https://drive.google.com/file/d/1D9gVR-FJtItIN82JD5-E__nHTAu8-beH/view

## Installation and Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the program:

```bash
python TIN.py
```

## Implementation Details

### Data Structure
- Nodes represent location IDs, edges represent trips between locations
- Interactions include source, destination, timestamp, and quantity information
- The program can process both the full dataset or a smaller test file

### Provenance Tracking
The program implements two strategies for tracking data provenance:

1. **LRB (Least Recently Born)**: In this strategy, data flows based on its creation time, with oldest data transferred first. This approach uses a minimum heap sorted by timestamp.

2. **MRB (Most Recently Born)**: This strategy prioritizes the newest data, transferring most recently generated data first. This is implemented using a maximum heap (inverted minimum heap).

Both strategies track where each unit of quantity originated from, allowing analysis of data flow patterns in the network.

### User Interface
The program offers an interactive command-line interface that allows users to:
- Select which dataset to use (test file or taxi data)
- Choose whether to visualize the network
- Select the provenance tracking strategy (LRB or MRB)
- Specify a target node or analyze all nodes
- Set a timestamp limit for analysis

## Data Format

TIN data is stored as a tab-separated text file with the format: `source_node destination_node timestamp flow_quantity`. The first two lines specify the number of vertices and edges respectively.

## References

[1]	C. K. N. Mamoulis, "Provenance in Temporal Interaction Networks," 2021, doi: 10.48550/arxiv.2110.05041.
