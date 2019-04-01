import sys
import numpy as np
sys.path.append("..")
import grading


def submit_cartpole(generate_session, email, token):
    sessions = [generate_session() for _ in range(100)]
    session_rewards = np.array(sessions)
    grader = grading.Grader("oyT3Bt7yEeeQvhJmhysb5g")
    grader.set_answer("7QKmA", int(np.mean(session_rewards)))
    grader.submit(email, token)


def submit_kungfu(session_rewards, email, token):
    grader = grading.Grader("6sPnVCn6EeieSRL7rCBNJA")
    # NOTE: Bug, so multiplied with 100, see
    #       https://www.coursera.org/learn/practical-rl/programming/95h2n/grader-a3c/discussions/threads/Q1XYCe08EeigTgpIFoGjKg
    #       https://www.coursera.org/learn/practical-rl/programming/95h2n/grader-a3c/discussions/threads/F1JHgGRmEei8iRK3RpIYPA/replies/vILx4WchEeimAgoMPemTPA
    grader.set_answer("HhNVX", 100*int(np.mean(session_rewards)))
    grader.submit(email, token)
