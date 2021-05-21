from xml.etree import ElementTree
from xml.dom import minidom

root = ElementTree.parse("DAPG_touchcopy.xml").getroot()
root.find('include').attrib["file"] = "objects/airplane.xml"
rawtext = ElementTree.tostring(root)
dom = minidom.parseString(rawtext)
with open("output.xml", "w") as f:
    dom.writexml(f, indent="\t", newl="", encoding="utf-8")