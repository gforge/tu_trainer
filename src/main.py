import traceback
from global_cfgs import Global_Cfgs
from scenario import Scenario
from UIs.console_UI import Console_UI


def main():
    cfgs = Global_Cfgs()
    scenario = Scenario(scenario_name=cfgs.get('scenario'))
    print(f'Running scenario {scenario.get_name()}')

    first = True
    scene = "None"
    try:
        for scene in iter(scenario):
            if cfgs.get('test_at_start', default=False) and first:
                # The learning must be initiated, is done again in the run_scene
                scene.update_learning_rate(epoch_no=0)
                scene.run_evaluation_and_test('@pre_run')
            first = False
            print(f'Running scene {scene.get_name()}')
            scene.run_scene()

    except RuntimeError as error:
        Console_UI().warn_user(error)
        Console_UI().inform_user("\n\n Traceback: \n")

        traceback.print_exc()
    except KeyboardInterrupt:
        Console_UI().inform_user(f'\nInterrupted by ctrl+c - stopped @ "{scene.get_name()}"')
    else:
        Console_UI().inform_user("Done with all scenarios!")

    Console_UI().inform_user('To view results, checkout the tensorboard:')
    Console_UI().inform_user(f'tensorboard --logdir /media/max/HD_1_3TB/log/{cfgs.sub_log_path}/tensorboard')


if __name__ == "__main__":
    main()
