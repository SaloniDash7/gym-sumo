Generating Intersections 
http://sumo.dlr.de/wiki/Tutorials/Quick_Start#Example_description

1. NODES  intersection.nod.xml 
2. EDGES lane1.edg.xml
3. ASSEMBLE lane1.netccfg
4. BUILD  lane1.net.xml

netconvert -c lane1.netccfg

5. SCENARIO left.rou.xml
6. ASSEMBLE left2.sumocfg
7. in python

i call it from python code just fine
sumoConfig = "roadNetwork/left2.sumocfg"

sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(PORT), \
	"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
	"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

