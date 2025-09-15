from neural_recommendation.applications.interfaces.dtos.filter_page import FilterPage
from neural_recommendation.applications.interfaces.dtos.movie import MovieList, MoviePublic
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository


class GetMoviesUseCase:
    def __init__(self, movie_repository: MovieRepository):
        self.movie_repository = movie_repository

    async def execute(self, filter_page: FilterPage) -> MovieList:
        if getattr(filter_page, "title", None):
            movies = await self.movie_repository.search_by_title(
                title=filter_page.title,
                offset=filter_page.offset,
                limit=filter_page.limit,
            )
        else:
            movies = await self.movie_repository.get_all(offset=filter_page.offset, limit=filter_page.limit)

        movie_publics = [MoviePublic(id=movie.id, title=movie.title, genres=movie.genres) for movie in movies]

        return MovieList(movies=movie_publics)
