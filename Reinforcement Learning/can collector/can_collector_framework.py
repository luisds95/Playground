'''
May 25, 2019.
Luis Da Silva.

Implements a playable version of the can collector robot game.
'''

import arcade
import numpy as np
import random
import time


class CanCollector(arcade.Window):
    """
    Main application class.
    """
    def __init__(self, row_count, column_count, width, height,
                 margin, screen_title, max_cans=10, seconds=1, max_steps=200):
        self.box_width = width
        self.box_height = height
        self.box_margin = margin
        self.screen_width = (width + margin) * column_count + margin
        self.screen_height = (height + margin) * row_count + margin + 30
        self.rows = row_count
        self.columns = column_count

        super().__init__(self.screen_width, self.screen_height, screen_title)
        arcade.set_background_color(arcade.color.WHITE)

        self.board = None

        self.player = None
        self.player_list = None
        self.can_list = None
        self.n_cans = 0
        self.max_cans = max_cans
        self.cans_recollected = 0
        self.recharge = None
        self.recharge_list = None
        self.dead = 0
        self.score = 0
        self.steps = 0
        self.step_time = None
        self.max_steps = max_steps
        self.battery_state = 'High'
        self.battery = 0

        self.game_over = False

        self.start_time = None
        self.min_seconds = seconds

        self.first_row = self.get_y(0)
        self.last_row = self.get_y(self.rows - 1)
        self.first_col = self.get_x(0)
        self.last_col = self.get_x(self.columns - 1)
        self.change_activated = 0

    def setup(self):
        # Set up board
        self.board = new_board(self.rows, self.columns)
        self.board_list = arcade.ShapeElementList()
        self.centers = []
        for row in range(len(self.board)):
            for column in range(len(self.board[0])):
                x = self.get_x(column)
                y = self.get_y(row)
                square = arcade.create_rectangle_filled(x, y, self.box_width, self.box_height, arcade.color.EGGSHELL)
                self.board_list.append(square)
                self.centers.append((x, y))

        # Set up player
        self.player_list = arcade.SpriteList()
        self.player = arcade.Sprite('icons/bot.png', 0.09)
        position = random.choice(self.centers)
        self.player.center_x = position[0]
        self.player.center_y = position[1]
        self.player_list.append(self.player)

        # Set up recharge station
        self.recharge_list = arcade.SpriteList()
        self.recharge = arcade.Sprite('icons/recharge.png', 0.09)
        self.recharge.center_x = position[0]
        self.recharge.center_y = position[1]
        self.recharge_list.append(self.recharge)

        # Set up battery
        self.reset_battery()
        self.dead = 0

        # Set up Cans
        self.can_list = arcade.SpriteList()
        self.n_cans = 0
        self.cans_recollected = 0
        self.add_can(5)

        # Score
        self.score = 0

        # Miscellaneous
        self.start_time = time.time()
        self.step_time = time.time()
        self.game_over = False
        self.steps = 0

    def get_x(self, column):
        return (self.box_margin + self.box_width) * column + self.box_margin + self.box_width // 2

    def get_y(self, row):
        return self.screen_height - (self.box_margin + self.box_height) * (row+1) + \
                                  self.box_margin + self.box_height // 2

    def add_can(self, n=1):
        positions = []
        for c in range(n):
            can = arcade.Sprite('icons/can.png', 0.09)
            while True:
                position = random.choice(self.centers)
                if position not in positions:
                    positions.append(position)
                    break
            can.center_x = position[0]
            can.center_y = position[1]
            self.can_list.append(can)
        self.n_cans += n

    def add_new_cans_with_time_check(self, n=1):
        if time.time() - self.start_time > np.random.normal(self.min_seconds, self.min_seconds/5):
            if self.n_cans < self.max_cans:
                self.add_can(n)
            self.start_time = time.time()

    def draw_game(self):
        # Draw sprites
        self.board_list.draw()
        self.can_list.draw()
        self.recharge_list.draw()
        self.player_list.draw()

        # Print score
        arcade.draw_text(f"Score: {self.score}", 10, 15, arcade.color.BLACK)
        arcade.draw_text(f"Recollected cans: {self.cans_recollected}", 80, 15, arcade.color.BLACK)
        color = arcade.color.GREEN if self.battery_state == 'High' else arcade.color.RED
        arcade.draw_text(f"Battery: {self.battery_state}", 225, 15, color)
        arcade.draw_text(f"Steps: {self.steps}", 315, 15, arcade.color.BLACK)
        if self.dead:
            arcade.draw_text(f"BATTERY DEAD", self.screen_width / 2, self.screen_height / 2, arcade.color.RED, 24,
                             bold=True, anchor_x='center')
            self.dead += 1

    def draw_game_over(self):
        arcade.draw_text(f"GAME OVER", self.screen_width / 2, self.screen_height - 60, arcade.color.RED, 40,
                         bold=True, anchor_x='center')
        arcade.draw_text(f"Score: {self.score}", self.screen_width / 2, self.screen_height / 2 + 34, arcade.color.BLACK,
                         24, anchor_x='center')
        arcade.draw_text(f"Recollected cans: {self.cans_recollected}", self.screen_width / 2,
                         self.screen_height / 2, arcade.color.BLACK, 24, anchor_x='center')
        arcade.draw_text(f"Steps: {self.steps}", self.screen_width / 2, self.screen_height / 2 - 34,
                         arcade.color.BLACK, 24, anchor_x='center')
        arcade.draw_text(f'Press "K" to restart', self.screen_width / 2, 30,
                         arcade.color.BLACK, 24, anchor_x='center')

    def on_draw(self):
        """
        Render the screen.
        """
        arcade.start_render()
        if not self.game_over:
            self.draw_game()
        else:
            self.draw_game_over()

    def update_game(self):
        """
                Collecting cans
                """
        self.can_list.update()
        if self.change_activated == 1:  # If a move was made
            self.player_list.update()
            self.change_activated = 2  # Change updated
            self.score -= 1
            self.steps += 1

            # Check for recharge:
            if arcade.check_for_collision(self.player, self.recharge):
                self.reset_battery()
            self.update_battery()

            self.step_time = time.time()
            if self.dead > 100:
                self.dead = 0

        elif time.time() - self.step_time > 2:
            self.steps += 1
            self.step_time = time.time()

        # Check for recollected cans
        can_hits = arcade.check_for_collision_with_list(self.player, self.can_list)
        for can in can_hits:
            can.kill()
            self.n_cans -= 1
            self.cans_recollected += 1
            self.score += 4

        # Add new cans
        self.add_new_cans_with_time_check()

        # Check game over
        if self.steps >= self.max_steps:
            self.game_over = True

    def update(self, delta_time):
        if self.game_over:
            pass
        else:
            self.update_game()

    def reset_battery(self):
        self.battery_state = 'High'
        self.battery = 60

    def update_battery(self):
        amount = random.randrange(0, 5)
        self.battery -= amount
        if self.battery <= 20:
            self.battery_state = 'Low'
        if self.battery <= 0:
            self.score -= 25
            self.reset_battery()
            self.player.center_x = self.recharge.center_x
            self.player.center_y = self.recharge.center_y
            self.dead = True

    def on_key_press(self, key, key_modifiers):
        """
        Called whenever a key on the keyboard is pressed.
        """
        if not self.game_over:
            if self.change_activated < 2:  # If key hasn't been released
                if key == arcade.key.LEFT and self.player.position[0] > self.first_col:
                    self.player.change_x = -(self.box_height + self.box_margin)
                    self.change_activated = 1  # Change applied
                elif key == arcade.key.RIGHT and self.player.position[0] < self.last_col:
                    self.player.change_x = (self.box_height + self.box_margin)
                    self.change_activated = 1
                elif key == arcade.key.UP and self.player.position[1] < self.first_row:
                    self.player.change_y = (self.box_width + self.box_margin)
                    self.change_activated = 1
                elif key == arcade.key.DOWN and self.player.position[1] > self.last_row:
                    self.player.change_y = -(self.box_width + self.box_margin)
                    self.change_activated = 1

        else:
            if key == arcade.key.K:
                self.setup()

    def on_key_release(self, key, key_modifiers):
        """
        Called whenever the user lets off a previously pressed key.
        """
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player.change_x = 0
        elif key == arcade.key.UP or key == arcade.key.DOWN:
            self.player.change_y = 0
        self.change_activated = 0  # Key released


def new_board(rows, columns):
    return [[0 for _ in range(columns)] for _ in range(rows)]


def run_game(row_count=10, column_count=10, width=50, height=50,
             margin=5, screen_title='Can Collector Robot'):

    game = CanCollector(row_count, column_count, width, height,
                        margin, screen_title)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    run_game()