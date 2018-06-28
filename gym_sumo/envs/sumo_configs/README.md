Generating Intersections 
http://sumo.dlr.de/wiki/Tutorials/Quick_Start#Example_description

netconvert -c E0.netccfg



1. set up the nodes

blah.nod.xml 
contains node names and x,y positions
<node id="91" x="-1000.0" y="1000.0" />

1b. optionally set up road conditions, this can also be specified in the egde file
blah.typ.xml
specifies roads conditions (priority & num lanes), this can also be set in edg.xml
<type id="a" priority="3" numLanes="3" speed="13.889"/> 

2. set up edges to link the nodes
blah.edg.xml
<edge id="L18" from="3" to="4" type="a"/>

2b. optionally set up conditions, like can't move from this lane to that lane
blah.con.xml
specifies connections, in the event that you want certain behaviors disallowed

3. write a file to assemble the separate parts and specify the name of the generated output (blah.net.xml)
blah.netccfg
assembles the other files and produces a blah.net.xml file. This completes the rode specification

4. build blah.net.xml
netconvert -c blah.netccfg

5. specify the traffic conditions: car types, routes, and starting points
blah.rou.xml
specifies traffic: car types, routes, origins, 
<vType accel="1.0" decel="5.0" id="CarD" length="7.5" minGap="2.5" maxSpeed="30.0" sigma="0.5" />
<route id="route01" edges="D2 L2 L12 L10 L7 D7"/>
<vehicle depart="54000" id="veh0" route="route01" type="CarA" color="1,0,0" /> 

6. assemble the rou and net files
blah.sumocfg
combine the net and rou files
also specify start and end times
and teleport time

7. this doesn't work for me, 
sumo –c quickstart.sumocfg 

i call it from python code just fine
sumoConfig = "roadNetwork/i01.sumocfg"

sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(PORT), \
	"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
	"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

