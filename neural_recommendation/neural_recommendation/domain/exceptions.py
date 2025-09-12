class DomainError(Exception):
    pass


class ValidationError(DomainError):
    pass


class NotFoundError(DomainError):
    pass


class RepositoryError(DomainError):
    pass


class ModelLoadError(DomainError):
    pass


class FeatureProcessingError(DomainError):
    pass


class AnnoyIndexError(DomainError):
    pass


class ConfigurationError(DomainError):
    pass


class ColdStartRecommendationError(DomainError):
    pass


class CandidateGeneratorError(DomainError):
    pass
