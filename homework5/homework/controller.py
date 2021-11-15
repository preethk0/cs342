import pystk
import math

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    target = 40
    if current_vel > target:
      action.acceleration = 0
      action.brake = True
    else:
      action.acceleration = -1 * ((current_vel - target) / target)
      action.brake = False

    angle = ((math.atan2(aim_point[1], aim_point[0]) * 180) / math.pi) + 90
    if angle < 0:
      angle = angle + 360

    if angle >= 0 and angle <= 180:
      dir = angle / 180
    else:
      dir = ((angle - 180) / 180) - 1

    action.steer = dir

    if -0.005 < dir < 0.005:
      action.nitro = True
      action.brake = False
    else:
      action.nitro = False

    if dir < -0.3 or dir > 0.3:
      action.drift = True
    else:
      action.drift = False

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
