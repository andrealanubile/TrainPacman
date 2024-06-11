import * as constants from './constants'

class Animator {
    constructor(frames=[], speed=20, loop=true) {
        this.frames = frames
        this.current_frame = 0
        this.speed = speed
        this.loop = loop
        this.dt = 0
        this.finished = false
    }

    reset() {
        this.current_frame = 0
        this.finished = false
    }

    update(dt) {
        if (!this.finished) {
            this.nextFrame(dt)
        }
        if (this.current_frame == this.frames.length) {
            if (this.loop) {
                this.current_frame = 0
            } else {
                this.finished = true
                this.current_frame -= 1
            }
        }
        return this.frames[this.current_frame]
    }

    nextFrame(dt) {
        this.dt += dt
        if (this.dt >= (1.0 / this.speed)) {
            this.current_frame += 1
            this.dt = 0
        }
    }
}

export default Animator;


