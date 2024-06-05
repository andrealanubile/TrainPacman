import * as constants from './constants'
import Spritesheet from './spritesheet';
import { Container, Sprite, spritesheetAsset } from 'pixi.js'

class LifeSprites extends Spritesheet {
    constructor(scaleX = 1, scaleY = 1) {
        super();
        this.images = [];
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.livesContainer = new Container();
    }

    removeImage() {
        if (this.images.length > 0) {
            this.images.shift();
        }
    }

    resetLives(numLives) {
        this.images = [];
        for (let i = 0; i < numLives; i++) {
            this.images.push(this.getImage(0, 0))
        }
    }

    getSprite() {
        return this.livesContainer;
    }

    getImage(x, y) {
        return super.getImage(Math.floor(x/2), Math.floor(y/2), 2*constants.TILEWIDTH, 2*constants.TILEHEIGHT);
    }

    drawLives() {
        this.livesContainer.removeChildren();
        for (let i = 0; i < this.images.length; i++) {
            let lifeSprite = new Sprite(this.images[i]);
            lifeSprite.x = i * constants.TILEWIDTH * 2 * this.scaleX;
            lifeSprite.y = (constants.SCREENHEIGHT - constants.TILEWIDTH * 2)*this.scaleY;
            lifeSprite.scale.set(this.scaleX, this.scaleY);
            this.livesContainer.addChild(lifeSprite);
        }
    }

    updateScale(scaleX, scaleY) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }

}

export default LifeSprites;