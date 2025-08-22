
Continuous intraday analysis
* Goal:
. Set up an updateable python script for KPIs of an analysis of the continuous intraday market
* Background:
. Beginning of 2026, the Intraday Cross-zonal Gate Closure Time (IDCZGC) will be set at 30mminutes
. We would like to understand, whether this has an impact on traded cross-border volumes
. Hypothesis is that we see a peak in traded cross-border volumes and that this peak shifts in response to the shortening the period between gate closure time & delivery
. In order to be able to judge its impact next year, we have to have a good understanding of how market participants are behaving at the moment. We therefore want to set the analysis up for an ex-post time horizon and update it in the coming events.

Data structure
* Capacity
.Traded cross-border capacity in the continuous intraday market & remaining capacity after allocation

* Balancing group/TSO/Direction
.Direction most important to us as we want to get neighboring regions

* Delivery:
. When is the power supposed to be delivered? -> 15.07.2025 00:00 means trade occurred for power being delivered at this time

* Allocation/Request time: 
. In contrast to delivery, it's the time stamp of the trade allocation
. Allocation type provides information whether capacity was allocated in DA market or intraday maeket.
. We are only interested in implicit allocations


Python Script
* Read in data
* Create the following charts:

-- KPI 1.1— Matched Capacity for implicit allocation per time to delivery start (Line chart) (absolute)
. Chart
	. Separate chart for each border
	. X-axis = "Time to delivery" => difference between "Delivery Start" and "Allocation time (in 5 min intervals) :
	. Y-axis = volume of matched capacity
	. Both directions in one chart (imports with minus sign)

-- KPI 1.2— Matched capacity per time to delivery start (Line chart) (relative)
. Chart
	. Separate chart for each border
	. X-axis ="Time to delivery" => difference between "Delivery Start" and "Allocation time (in 5 min intervals)
	. Y-axis = relative volume of matched capacity (matched capacity for a time to delivery divided by the total matched capacity per day)
	. Both directions in one chart (imports with minus sign)

. Statistics 
	. Average, min, max, quartiles for relative matched capacity per time to delivery


--KPI 1.3 — Matched capacity for implicit allocation per time to delivery start (Linechart)
. Chart 
	. Separate chart for each border
	. Line legend for allocation time
.Statistics
	.1)Calculate the sum of matched capacity per delivery start 2) divide the matched capacity of each allocation time by sum of capacity per delivery start 3) Calculate the average per allocation time

-- KPI 1.4 - Matched capacity per implicit allocation per allocation time (Line chart)
. Chart 
	. Separate chart for each border
	. Line legend for allocation time
	. Aggregate per delivery minute or every 5 minutes


-- KPI2— Available capacity per implicit allocation per border until delivery?

. ATC after allocation for delivery start

. Separate chart for every border

-- KPI 3.1 - When is max cross-border capacity for implicit allocations utilized?

. Chart with max capacity and total sum of matched capacity per delivery start per border (i.e. sum over all allocation times)
. Aggregate per delivery minute or every 5 minutes

-- KPI 3.2 - How often and how long is the full cross-border capacity for implicit allocations utilized?
. Count of single and continuous hours where total sum of matched capacity per delivery start reaches max capacity per border.

