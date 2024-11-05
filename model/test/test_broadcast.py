import pypose as pp
import torch

def test_multi_pose_action():
    tf = pp.SE3([[-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503],
                 [-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503]])
    pc_multi = torch.rand(2, 3)

    

    broadcasted = tf.Act(pc_multi)

    assert(torch.equal(broadcasted[0,:], tf[0,:] @ pc_multi[0,:]))
    assert(torch.equal(broadcasted[1,:], tf[1,:] @ pc_multi[1,:]))
    print(tf[1,:] @ pc_multi[1,:])
    print(broadcasted[1,:])

def test_bmv():
    mat = torch.randn(2, 2, 3, 3)
    vec = torch.randn(2, 4, 3)
    out = pp.bmv(mat, vec)
    print(out.shape)
    
def test_multi_pose_action2():
    tf = pp.SE3([[-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503],
                 [-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503]])
    tf = tf.unsqueeze(1)
    print("tf after unsqueeze ", tf)
    tf = tf.expand(2,4,7)
    print("tf after expand ", tf)
    pc_multi = torch.rand(2, 4, 3)

    print(tf[0] @ pc_multi[0, :])
    
    broadcasted = tf.Act(pc_multi)

    assert(torch.equal(broadcasted[0,:], tf[0,:] @ pc_multi[0,:]))
    assert(torch.equal(broadcasted[1,:], tf[1,:] @ pc_multi[1,:]))
    print(tf[1,:] @ pc_multi[1,:])
    print(broadcasted[1,:])
    

if __name__ == "__main__":
    print(test_multi_pose_action2())
