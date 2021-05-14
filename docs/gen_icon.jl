using Luxor

origin = Point(0, 0)
scale = 1
r = 90 * scale
c1 = Point(0, -r) + origin
c2 = Point(-r * 3^0.5 / 2, r / 2) + origin
c3 = Point(0, r) + origin
c4 = Point(r * 3^0.5 / 2, r / 2) + origin
box_size = 15 * scale
circle_size = 30 * scale

box_node1 = Point(0, -box_size) + origin
box_node2 = Point(-box_size, box_size) + origin
box_node3 = Point(0, box_size) + origin
box_node4 = Point(box_size, box_size) + origin

c_node1 = c1 + Point(0, circle_size)
c_node2 = c2 + Point(circle_size * 3^0.5 / 2, -circle_size / 2)
c_node3 = c3 + Point(0, -circle_size)
c_node4 = c4 + Point(-circle_size * 3^0.5 / 2, -circle_size / 2)

arc_node1 = box_node2 + (c_node2 - box_node2) * 0.2
arc_node2 = box_node4 + (c_node4 - box_node4) * 0.2

arrow_width = 5 * scale
aha = pi / 4

Drawing(320, 320, "docs/src/assets/logo.svg")
background(1, 1, 1, 0)
Luxor.origin()

setcolor(0.0, 0.0, 0.0) # black
box(Point(-box_size, -box_size) + origin, Point(box_size, box_size) + origin, :fill)
arc2r(origin, arc_node2, arc_node1, :stroke)
setcolor(0.251, 0.388, 0.847)  # dark blue
circle(c1, circle_size, :fill)
arrow(c_node1, box_node1, linewidth = arrow_width, arrowheadangle = aha)
setcolor(0.796, 0.235, 0.2)  # dark red
circle(c2, circle_size, :fill)
arrow(box_node2, c_node2, linewidth = arrow_width, arrowheadangle = aha)
setcolor(0.22, 0.596, 0.149) # dark green
circle(c3, circle_size, :fill)
arrow(box_node3, c_node3, linewidth = arrow_width, arrowheadangle = aha)
setcolor(0.584, 0.345, 0.698) # dark purple
circle(c4, circle_size, :fill)
arrow(box_node4, c_node4, linewidth = arrow_width, arrowheadangle = aha)

finish()
