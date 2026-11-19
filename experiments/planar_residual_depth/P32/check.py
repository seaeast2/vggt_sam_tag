"""Deterministic checks for spatial filtering, visibility and loop optimization."""
import cv2
import numpy as np

from matching import transform
from replay import optimize
from spatial import PatchIndex, project_masks


def check_index():
    rng = np.random.default_rng(32)
    xy = rng.uniform(-20, 20, (240,3))
    extents = rng.uniform(.01, 3., xy.shape)
    patches = [dict(id=i, bounds=[center-size, center+size])
               for i,(center,size) in enumerate(zip(xy,extents,strict=True))]
    linear, bvh, bsp = [PatchIndex(patches, kind) for kind in ('linear','bvh','bsp')]
    assert bvh.stats()['nodes'] > 1 and bsp.stats()['nodes'] > 1
    k = np.array([[150.,0.,80.],[0.,150.,60.],[0.,0.,1.]])
    for _ in range(30):
        r = cv2.Rodrigues(rng.normal(size=3))[0]
        projection = k @ np.column_stack((r,rng.normal(size=3)))
        for margin in (0,12):
            expected = linear.query(projection,(120,160),margin)
            assert bvh.query(projection,(120,160),margin) == expected
            assert bsp.query(projection,(120,160),margin) == expected
    assert not PatchIndex([]).query(np.column_stack((k,np.zeros(3))),(120,160))


def check_footprints():
    camera = np.column_stack((np.eye(3),np.zeros(3)))

    def square(x1,y1,x2,y2,z,parent=-1):
        return dict(parent=parent,xyz=[[x*z,y*z,z] for x,y in
                     [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]])

    outlines = {1:[dict(plane=[0,0,1,1],rings=[
                    square(5,5,35,35,1), square(13,13,27,27,1,0),
                    square(42,7,49,14,1)])],
                2:[dict(plane=[0,0,1,2],rings=[square(5,5,35,35,2)])]}
    points = {i: np.array([[0.,0.,i]]) for i in outlines}
    result = project_masks(points,camera,(55,55),[1,2],footprints=outlines)
    assert result[1][8,8] and result[1][11,46] and not result[1][20,20]
    assert result[2][20,20] and not result[2][8,8]


def check_graph():
    poses = np.stack([transform(np.eye(3),[x,0,0]) for x in (0,1.1,2.2,1.3,.4)])
    z = np.zeros(3)
    edges = [(i,i+1,np.linalg.inv(poses[i+1]) @ poses[i]) for i in range(4)]
    result = optimize(poses,edges+[(0,4,transform(np.eye(3),z))])
    np.testing.assert_array_equal(result[0],poses[0])
    assert np.linalg.norm(result[-1,:3,3]) < .36


if __name__ == '__main__':
    check_index()
    check_footprints()
    check_graph()
    print('Frustum, footprint, occlusion and pose graph checks passed.')
