class ModelBasedReflexAgent:
    def __init__(self, desired_temperature):
        self.desired_temperature = desired_temperature
        self.previous_action = None

    def percieve(self, current_temperature):
        return current_temperature
    
    def act(self, current_temperature):
        if current_temperature < self.desired_temperature:
            if self.previous_action != "Turn on heater":
                action = "Turn on the heater"
            else:
                action = "Heater is already ON!"
        else:
            if self.previous_action != " Turn off the heater!":
                action = "Turn off the heater"
            else:
                action = "Heater is already OFF!"

        self.previous_action = action
        return action
    
rooms = {
    "Living Room": 18,
    "Bedroom" : 20,
    "Kitchen" : 27,
    "Bathroom" : 19
}
desired_temperature = 23

agent = ModelBasedReflexAgent(desired_temperature)
for room, temperature in rooms.items():
    action = agent.act(temperature)
    print(f"{room}: Current Temperature = {temperature}C.{action}")
    
            