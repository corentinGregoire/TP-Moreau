class Config:
    DEBUG = False
    DEVELOPMENT = False
    PROPAGATE_EXCEPTIONS = False

    X_API_KEY = "08e869be-8f84-450c-a775-954c91c0581a"


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True
    DEVELOPMENT = True
