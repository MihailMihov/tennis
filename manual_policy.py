import pygame

class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

        # TO-DO: show current agent observation if this is True
        self.show_obs = show_obs

        # action mappings for all agents are the same
        if True:
            self.default_action = 0
            self.action_mapping = dict()
            self.action_mapping[pygame.K_SPACE] = 1  # weapon
            self.action_mapping[pygame.K_w] = 2  # move up
            self.action_mapping[pygame.K_d] = 3  # move right
            self.action_mapping[pygame.K_a] = 4  # move left
            self.action_mapping[pygame.K_s] = 5  # move down

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        if not pygame.get_init():
            pygame.init()

        # if we get a key, override action using the dict
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # escape to end
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    # backspace to reset
                    self.env.reset()

                elif event.key in self.action_mapping:
                    action = self.action_mapping[event.key]

        pressed = pygame.key.get_pressed()
        for key, key_action in self.action_mapping.items():
            if pressed[key]:
                return key_action

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping

