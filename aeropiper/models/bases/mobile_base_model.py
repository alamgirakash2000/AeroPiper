"""
Defines the mobile base model
"""

from aeropiper.models.bases.robot_base_model import RobotBaseModel


class MobileBaseModel(RobotBaseModel):
    @property
    def naming_prefix(self):
        return "mobilebase{}_".format(self.idn)
