import * as constants from './constants'
import Animator from './animator'
import Spritesheet from './spritesheet';
import { Sprite } from 'pixi.js'

class PacmanSprites extends Spritesheet {
    constructor() {
        super();
        this.alive = true;
        this.direction = constants.LEFT;
        this.image = this.getStartImage();
        this.animations = new Object()
        this.animations.left = new Animator([[8,0], [0, 0], [0, 2], [0, 0]])
        this.animations.right = new Animator([[10,0], [2, 0], [2, 2], [2, 0]])
        this.animations.up = new Animator([[10,2], [6, 0], [6, 2], [6, 0]])
        this.animations.down = new Animator([[8,2], [4, 0], [4, 2], [4, 0]])
        this.animations.death = new Animator([[0, 12], [2, 12], [4, 12], [6, 12], [8, 12], [10, 12], [12, 12], [14, 12], [16, 12], [18, 12], [20, 12]], 6, false)
    }

    getImage(loc) {
        return super.getImage(Math.floor(loc[0]/2), Math.floor(loc[1]/2), 2*constants.TILEWIDTH, 2*constants.TILEHEIGHT);
    }

    getStartImage() {
        return this.getImage([8, 0])
    }

    drawPacman(dt) {
        if (this.alive) {
            switch (this.direction) {
                case constants.LEFT:
                    this.image = this.getImage(this.animations.left.update(dt));
                    this.stopimage = [8, 0];
                    break;
                case constants.RIGHT:
                    this.image = this.getImage(this.animations.right.update(dt));
                    this.stopimage = [10, 0];
                    break;
                case constants.DOWN:
                    this.image = this.getImage(this.animations.down.update(dt));
                    this.stopimage = [8, 2]
                    break;
                case constants.UP:
                    this.image = this.getImage(this.animations.up.update(dt));
                    this.stopimage = [10, 2]
                    break;
                case constants.STOP:
                    this.image = this.getImage(this.stopimage);
                    
            }
        } else {
            this.image = this.getImage(this.animations.death);
        }

        let col = 13.5;
        let row = 26;
        let pacmanSprite = new Sprite(this.image);
        pacmanSprite.anchor.set(0.5);
        pacmanSprite.x = (col + 1/2) * constants.TILEWIDTH;
        pacmanSprite.y = (row + 1/2) * constants.TILEHEIGHT;;
        return pacmanSprite;
    }

}

export default PacmanSprites;