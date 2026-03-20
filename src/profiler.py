import cProfile
import pstats

from text8_example import text8_example

if __name__ == "__main__":
    profiler = cProfile.Profile()

    profiler.enable()
    text8_example()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)